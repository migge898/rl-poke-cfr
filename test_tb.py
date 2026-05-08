from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs/test")
for i in range(10):
    writer.add_scalar("Test/Value", i*i, i)
writer.close()
print("Test data written to logs/test")