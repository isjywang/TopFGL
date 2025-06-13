import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random

def run_local(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):

    print("begin local train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
   
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        
        for client in clients:
            client.local_train(local_epoch)

        if c_round == COMMUNICATION_ROUNDS:
            all_acc = 0
            all_size = 0
            for client in clients:
                acc = client.evaluate()
                all_acc+=acc*client.train_size
                all_size+=client.train_size
            print("average acc: ",all_acc/float(all_size))    

    print("train finished!")

    
def run_fedavg(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):
    print("begin fedavg train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients:
        client.download_from_server(args, server)

    all_train_time = 0
    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        begin = time.time()
        for client in clients:
            client.local_train(local_epoch)
        end = time.time()
        all_train_time = all_train_time + end - begin
           
        server.aggregate_weights(clients)
        for client in clients:
            client.download_from_server(args, server)

        all_acc = 0
        all_size = 0
        for client in clients:
            acc = client.evaluate()
            all_acc+=acc*client.train_size
            all_size+=client.train_size
        print("average acc: ",all_acc/float(all_size))
    print("train finished!")
    print("all train time:",all_train_time)



def run_topfgl(args, clients, server, COMMUNICATION_ROUNDS, local_epoch):

    print("begin TopFGL train")
    print("all communication rounds: ",COMMUNICATION_ROUNDS)
    print("local training epoch: ",local_epoch)
    for client in clients:
        client.download_from_server(args, server)

    for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        if (c_round) % 50 == 0:
            print(f"  > round {c_round}")
        for client in clients:
            client.local_train(local_epoch)
                
        if c_round < int(args.warm_epoch*COMMUNICATION_ROUNDS):
            server.aggregate_weights(clients)
            for client in clients:
                client.download_from_server(args, server)
        else:
            server.topfgl_aggregate(clients)
            for client in clients:
                client.download_from_server_gr(args, server)

        all_acc, all_size = 0, 0
        for client in clients:
            acc = client.evaluate()
            all_acc+=acc*client.train_size
            all_size+=client.train_size
        print("average acc: ",all_acc/float(all_size))

        if c_round>1 and c_round<COMMUNICATION_ROUNDS and c_round % int(1/float(1+args.repair_fre) *COMMUNICATION_ROUNDS) == 0:
            for client in clients:
                client.repair_subgraph()
            print("aug. finished")
            
    print("TopFGL train finished!")

