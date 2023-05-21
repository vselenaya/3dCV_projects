import socket
import time

num_packets = 20
len_packet = 50000
HOST = "127.0.0.1"
PORT = 123

udp_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
udp_server_socket.bind((HOST, PORT))
udp_server_socket.settimeout(2)

received_packets = 0
length_data = 0
all_time = 0
for i in range(num_packets):
    try:
        packet, _ = udp_server_socket.recvfrom(len_packet + 100)
    except socket.timeout:
        print("!")
        continue

    # print("Получен пакет: ", packet.decode('utf-8'))
    received_packets += 1
    length_data += len(packet)
    delta = time.time() - float(packet.decode('utf-8').split('#')[0])
    print(delta, time.time())
    all_time += delta
    time.sleep(3 - delta)

udp_server_socket.close()

print("Получено пакетов: {} из {}".format(received_packets, num_packets))
print("Средняя скорость передачи: {} байт в секунду".format(length_data / all_time))


