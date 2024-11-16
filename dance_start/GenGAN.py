import numpy as np
import cv2
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
from GenVanillaNN import VideoSkeletonDataset, init_weights, GenNNSkeToImage


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Base number of filters (64 is standard for DCGAN)
        ndf = 64

        self.main = nn.Sequential(
            # Layer 1: Input image (3 channels x 64x64)
            # Conv2d params: in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1
            # Output: (64 x 32 x 32)
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # Allows small negative values (better gradients)
            nn.Dropout2d(0.25),  # Prevents discriminator from being too strong

            # Layer 2: Input (64 x 32 x 32)
            # Output: (128 x 16 x 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),  # Normalizes activations for stable training
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Layer 3: Input (128 x 16 x 16)
            # Output: (256 x 8 x 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # Layer 4: Input (256 x 8 x 8)
            # Final conv layer to produce 1-channel output
            # Output: (1 x 4 x 4)
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()  # Squashes output to [0,1] range (probability of being real)
        )
        self.apply(init_weights)
        print(self.main)

    def forward(self, input):
        # Pass input through the network
        return self.main(input)


class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        # Initialize Generator and Discriminator networks
        self.netG = GenNNSkeToImage()  # Generator: skeleton -> image
        self.netD = Discriminator()  # Discriminator: image -> real/fake
        self.filename = 'data/Dance/DanceGenGAN.pth'  # Path to save/load model

        # Define image transformations for training
        tgt_transform = transforms.Compose([
            transforms.Resize(64),  # Resize images to 64x64
            transforms.CenterCrop(64),  # Ensure square aspect ratio
            transforms.ToTensor(),  # Convert to tensor (0-1 range)
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

        # Create dataset from video with skeleton pairs
        self.dataset = VideoSkeletonDataset(
            videoSke,  # Video with extracted skeletons
            ske_reduced=True,  # Use reduced skeleton (13 joints)
            target_transform=tgt_transform  # Apply transforms to target images
        )

        # Setup data loader for efficient batch training
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=16,  # Number of samples per batch
            shuffle=True,  # Randomize training samples
            num_workers=2,  # Parallel data loading processes
            pin_memory=True,  # Speed up GPU transfer
            persistent_workers=True  # Keep workers alive between epochs
        )

        # Setup GPU/CPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Move models to GPU if available
        if torch.cuda.is_available():
            print("CUDA available. Using GPU for training.")
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()


        # Load pretrained model if requested and exists
        if loadFromFile and os.path.isfile(self.filename):
            checkpoint = torch.load(self.filename)
            self.netG.load_state_dict(checkpoint['generator_state_dict'])
            self.netD.load_state_dict(checkpoint['discriminator_state_dict'])

    def train(self, num_epochs=200):
        # Define loss functions
        criterion_adv = nn.BCELoss()  # Binary Cross Entropy for adversarial loss
        criterion_l1 = nn.L1Loss()  # L1 loss for image reconstruction

        # Setup optimizers with different learning rates for G and D
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Learning rate schedulers - reduce LR when loss plateaus
        schedulerD = optim.lr_scheduler.ReduceLROnPlateau(
            optimizerD, mode='min', factor=0.5, patience=15, verbose=True
        )
        schedulerG = optim.lr_scheduler.ReduceLROnPlateau(
            optimizerG, mode='min', factor=0.5, patience=15, verbose=True
        )

        print("Starting Training Loop...")
        try:
            for epoch in range(num_epochs):
                # Initialize epoch statistics
                errD_epoch = 0.0  # Discriminator loss
                errG_epoch = 0.0  # Generator loss
                D_x_epoch = 0.0  # D(x) - real images score
                D_G_z_epoch = 0.0  # D(G(z)) - fake images score
                n_batch = 0

                for i, (skeletons, real_images) in enumerate(self.dataloader):
                    batch_size = real_images.size(0)
                    # Move data to GPU/CPU
                    real_images = real_images.to(self.device)
                    skeletons = skeletons.to(self.device)

                    # Create soft labels with noise for more stable training
                    real_label = (torch.rand(batch_size, 1, 4, 4) * 0.15 + 0.85).to(self.device)  # 0.85-1.0
                    fake_label = (torch.rand(batch_size, 1, 4, 4) * 0.15).to(self.device)  # 0.0-0.15

                    ############################
                    # Train Discriminator
                    ############################
                    self.netD.zero_grad()
                    # Train on real images
                    output_real = self.netD(real_images)
                    errD_real = criterion_adv(output_real, real_label)
                    D_x = output_real.mean().item()

                    # Train on fake images
                    fake_images = self.netG(skeletons)
                    output_fake = self.netD(fake_images.detach()) # detach to avoid G update
                    errD_fake = criterion_adv(output_fake, fake_label)
                    D_G_z1 = output_fake.mean().item()

                    # Combined D loss
                    errD = errD_real + errD_fake

                    # Only update D if it's too weak
                    if D_x < 0.65 or D_G_z1 > 0.35:  # Prevents D from overpowering G
                        errD.backward()
                        optimizerD.step()

                    ############################
                    # Update Generator
                    ############################
                    self.netG.zero_grad()
                    output_fake = self.netD(fake_images)  # Recompute D(G(z))

                    # Combine adversarial and L1 losses
                    errG_adv = criterion_adv(output_fake, real_label)  # Fool the discriminator
                    errG_L1 = criterion_l1(fake_images, real_images)  # Match real image

                    # Dynamic L1 weight - starts high, gradually decreases
                    l1_weight = max(30, 100 - epoch // 4)
                    errG = errG_adv + l1_weight * errG_L1
                    errG.backward()
                    D_G_z2 = output_fake.mean().item()
                    optimizerG.step()

                    # Collect statistics for logging
                    errD_epoch += errD.item()
                    errG_epoch += errG.item()
                    D_x_epoch += D_x
                    D_G_z_epoch += (D_G_z1 + D_G_z2) / 2
                    n_batch += 1

                    # Display progress every 32 batches
                    if i % 32 == 0:
                        print(f'[{epoch}/{num_epochs}][{i}/{len(self.dataloader)}] '
                              f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                              f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} '
                              f'L1_weight: {l1_weight}')

                        # Show current results
                        with torch.no_grad():
                            fake_img = self.dataset.tensor2image(fake_images[0].cpu())
                            real_img = self.dataset.tensor2image(real_images[0].cpu())
                            comparison = np.hstack([real_img, fake_img])
                            comparison = cv2.resize(comparison, (512, 256))
                            cv2.imshow('Real vs Fake', comparison)
                            cv2.waitKey(1)

                # Calculate epoch averages and update schedulers
                avg_errD = errD_epoch / n_batch
                avg_errG = errG_epoch / n_batch
                avg_D_x = D_x_epoch / n_batch
                avg_D_G_z = D_G_z_epoch / n_batch


                # Adjust learning rates if needed
                schedulerD.step(avg_errD)
                schedulerG.step(avg_errG)

                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    print(f"Saving checkpoint at epoch {epoch + 1}")
                    torch.save({
                        'epoch': epoch,
                        'generator_state_dict': self.netG.state_dict(),
                        'discriminator_state_dict': self.netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'schedulerG_state_dict': schedulerG.state_dict(),
                        'schedulerD_state_dict': schedulerD.state_dict(),
                    }, self.filename)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            cv2.destroyAllWindows()

    def generate(self, ske):
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(self.device)

        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(ske_t_batch)
            fake = fake.cpu()
            generated_image = self.dataset.tensor2image(fake[0])

        return generated_image


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory =", os.getcwd())
    print("GenGAN: Filename =", filename)

    try:
        # Check if video file exists
        if not os.path.exists(filename):
            print(f"Error: Video file '{filename}' not found")
            sys.exit(1)

        targetVideoSke = VideoSkeleton(filename)
        gen = GenGAN(targetVideoSke, loadFromFile=False)

        try:
            print("\nStarting training (press Ctrl+C to stop)...")
            gen.train(200)

            print("\nTesting generation (press 'q' to quit)...")
            for i in range(targetVideoSke.skeCount()):
                image = gen.generate(targetVideoSke.ske[i])
                image = cv2.resize(image, (256, 256))

                # Add frame counter
                cv2.putText(image, f"Frame: {i}/{targetVideoSke.skeCount()}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)

                cv2.imshow('Generated Image', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nGeneration stopped by user")
                    break

        except KeyboardInterrupt:
            print("\nProcess interrupted by user")

    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        cv2.destroyAllWindows()