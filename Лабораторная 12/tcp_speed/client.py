import socket
import time
import random

num_packets = 20  # число пакетов, которые будем отправлять
len_packet = 100000  # длина пакета (большая, чтобы время точнее измерялось)
HOST = "127.0.0.1"
PORT = 123

# символы, из которых собираем пакеты:
symbols = "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect((HOST, PORT))

    for i in range(num_packets):
        packet = str(time.time()) + '#'
        for j in range(len_packet):
            packet += random.choice(symbols)
        sock.sendall(packet.encode("utf-8"))

        time.sleep(3)

