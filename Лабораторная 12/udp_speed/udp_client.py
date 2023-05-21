import socket
import time
import random

num_packets = 20
len_packet = 50000
HOST = "127.0.0.1"
PORT = 123

symbols = "!#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    for i in range(num_packets):
        packet = str(time.time()) + '#'
        for j in range(len_packet):
            packet += random.choice(symbols)
        sock.sendto(packet.encode("utf-8"), (HOST, PORT))

        time.sleep(3)

