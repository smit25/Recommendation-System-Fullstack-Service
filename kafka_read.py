import paramiko
import socket
import select
import threading
from kafka import KafkaConsumer
import time
from tqdm import tqdm

def forward_tunnel(local_port, remote_host, remote_port, transport):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('', local_port))
        listener.listen(5)

        while transport.is_active():
            client_sock, client_addr = listener.accept()
            print(f"Accepted connection from {client_addr}")
            
            try:
                chan = transport.open_channel(
                    'direct-tcpip',
                    (remote_host, remote_port),
                    client_addr
                )
            except Exception as e:
                print(f"SSH forward failed: {e}")
                client_sock.close()
                continue

            threading.Thread(
                target=pipe_data,
                args=(client_sock, chan),
                daemon=True
            ).start()
    finally:
        listener.close()

def pipe_data(sock, chan):
    while True:
        r, _, _ = select.select([sock, chan], [], [])
        if sock in r:
            data = sock.recv(1024)
            if not data: break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if not data: break
            sock.send(data)
    sock.close()
    chan.close()

def connect_to_kafka_stream():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(
            hostname="128.2.204.215",
            username="tunnel",
            password="mlip-kafka"
        )
        print("SSH connection established")
        
        threading.Thread(
            target=forward_tunnel,
            args=(9092, 'localhost', 9092, ssh.get_transport()),
            daemon=True 
        ).start()
        
        print("Port forwarding active on localhost:9092")
        
        read_kafka_stream()
        
    finally:
        ssh.close()
        print("Connection closed")

def read_kafka_stream():
    bootstrap_servers = ['localhost:9092']
    topic = 'movielog23'
    group_id = 'movielog23_final_1'
    
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset='earliest'
    )
    
    output_file = 'kafka_stream_ingest.txt'
    cutoff_timestamp = int(time.time() * 1000)
    print(f"Cutoff timestamp (ms): {cutoff_timestamp}")

    finished_partitions = set()
    message_count = 0 

    with open(output_file, 'w') as f:
        with tqdm(desc="Processing Messages", unit="msg", dynamic_ncols=True) as pbar:
            while True:
                messages = consumer.poll(timeout_ms=1000)
                if not messages:
                    time.sleep(1)
                    continue
                
                for tp, partition_messages in messages.items():
                    if tp in finished_partitions:
                        continue
                    
                    for message in partition_messages:
                        if message.timestamp <= cutoff_timestamp:
                            f.write(f"{message.value.decode('utf-8')}\n")
                            message_count += 1
                            pbar.update(1)
                        else:
                            finished_partitions.add(tp)
                            print(f"Partition {tp} finished at offset {message.offset}")
                            break
                
                if finished_partitions == consumer.assignment():
                    break

    consumer.close()
    print(f"Finished consuming {message_count} messages. Output written to {output_file}")

if __name__ == "__main__":
    connect_to_kafka_stream()
    