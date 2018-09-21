import torch
# Optimizers
def Get_optimizers(args, generator, discriminator):
    optimizer_G = torch.optim.Adam(
                    generator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(
                    discriminator.parameters(),
                    lr=args.lr, betas=(args.b1, args.b2))

    return optimizer_G, optimizer_D
# Loss functions
def Get_loss_func(args):
    criterion_GAN = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    if torch.cuda.is_available():
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()
    return criterion_GAN, criterion_pixelwise
