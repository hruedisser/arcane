def lambda_scheduler(epoch):
    if epoch < 10:
        return 1e-4 + epoch * (1e-3 - 1e-4) / 10
    if epoch in range(10, 15):
        return 1e-3
    if epoch in range(15, 20):
        return 1e-4
    if epoch >= 20:
        return 1e-5
