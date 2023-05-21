import socket
import time

num_packets = 20
len_packet = 100000
HOST = "127.0.0.1"
PORT = 123

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("Ожидается подключение...")
connection, client_address = sock.accept()
print("... подключено")
connection.settimeout(2)

received_packets = 0
length_data = 0
all_time = 0
for i in range(num_packets):
    try:
        packet = connection.recv(len_packet + 1000)
    except socket.timeout:
        print("!")
        continue

    #print("Получен пакет: ", packet.decode('utf-8'))
    received_packets += 1
    length_data += len(packet)
    delta = time.time() - float(packet.decode('utf-8').split('#')[0])  # время на отправку пакетов
    print(delta, time.time())
    all_time += delta
    time.sleep(3 - delta)  # дожидаемся

connection.close()

print("Получено пакетов: {} из {}".format(received_packets, num_packets))
print("Средняя скорость передачи: {} байт в секунду".format(length_data / all_time))


