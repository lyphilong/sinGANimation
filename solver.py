import torch
import torch.nn.functional as F
from tensorboard import summary
from torchvision.utils import save_image


from model import Generator, Discriminator
from utils import Utils

import numpy as np

import os
import time
import datetime
import random
import glob




class Solver(Utils):

    def __init__(self, data_loader, config_dict):
        # NOTE: the following line create new class arguments with the
        # values in config_dict
        self.__dict__.update(**config_dict)

        self.data_loader = data_loader

        self.device = 'cuda:' + \
            str(self.gpu_id) if torch.cuda.is_available() else 'cpu'
        print(f"Model running on {self.device}")

        if self.use_tensorboard:
            self.build_tensorboard()

        self.loss_visualization = {}

        self.build_model()




    def train(self):
        print('Training...')

        self.global_counter = 0
        self.first_scale = 0
        self.num_scale = 1
        self.num_scale_prev = 1

        if self.resume_iters:
            self.first_iteration = self.resume_iters
            self.restore_model(self.resume_iters)
        else:
            self.first_iteration = 0

        self.start_time = time.time()
        self.alpha_rec = 1
        self.get_training_data()

        for scale in range(self.first_scale, self.num_scale):

            for iteration in range(1000):


                self.iteration = iteration

                self.train_discriminator()
                #đầu tiên train discriminator

                if (self.iteration+1) % self.n_critic == 0:
                    generation_outputs = self.train_generator()


                if (self.iteration+1) % self.sample_step == 0:
                    self.print_generations(generation_outputs)

                if self.iteration % self.model_save_step == 0:
                    self.save_models(self.iteration, self.epoch)

                if self.iteration % self.log_step == 0:
                    self.update_tensorboard()
                    self.global_counter += 1

            # Decay learning rates. , giảm learning rate, giảm đi 0.1
            if (self.iteration+1) > self.num_epochs_decay:
                # float(self.num_epochs_decay))
                self.g_lr -= (self.g_lr / 10.0)
                # float(self.num_epochs_decay))
                self.d_lr -= (self.d_lr / 10.0)
                self.update_lr(self.g_lr, self.d_lr)
                print('Decayed learning rates, self.g_lr: {}, self.d_lr: {}.'.format(
                    self.g_lr, self.d_lr))

            # Save the last model, lưu model cuối cùng lại
            '''
            TO DO: 
            
            '''
            self.num_scale_prev = self.num_scale
            self.save_models()
            '''
            TODO: 
            upscale của hình tái cấu trúc lên trước khi bỏ vào x_real
            Sau đó, rest các tham số train về lại từ đầu 
            '''



    def get_training_data(self):

        self.x_real, self.label_org = self.data_loader.dataset[0]
#        print(self.x_real.size())
#        print(self.label_org)
        self.x_real = self.x_real.view(-1,3,128,128)
        self.x_real = self.x_real.to(self.device)
        # Lấy hình ảnh thật
        # Labels for computing classification loss.
        self.label_org = self.label_org.to(self.device)
        #Tính AUs gốc

        # Get random targets for training
        self.label_trg = self.get_random_labels_list()
        # Lấy ngẫu nhiên AUs mong muốn nào cũng được
        self.label_trg = torch.FloatTensor(self.label_trg).clamp(0, 1)
        # Labels for computing classification loss.
        self.label_trg = self.label_trg.to(self.device)

        if self.use_virtual:
            self.label_trg_virtual = self.get_random_labels_list()
            self.label_trg_virtual = torch.FloatTensor(
                self.label_trg_virtual).clamp(0, 1)
            # Labels for computing classification loss.
            self.label_trg_virtual = self.label_trg_virtual.to(self.device)

            assert not torch.equal(
                self.label_trg_virtual, self.label_trg), "Target label and virtual label are the same"

    def get_random_labels_list(self):
        trg_list = []
        trg_list_aux = self.data_loader.dataset[1][1]
        trg_list.append(trg_list_aux.numpy() +
                        np.random.uniform(-0.1, 0.1, trg_list_aux.shape))
        return trg_list

    def train_discriminator(self):
        # Compute loss with real images.

        if(self.num_scale !=self.num_scale_prev):
            self.x_real = self.imsize(self.x_real,3/4)

        critic_output, classification_output = self.D(self.x_real)
        #lấy hai kết quả một cái của D_I và D_Y ra
        d_loss_critic_real = -torch.mean(critic_output)
        # tính loss cho d
        d_loss_classification = torch.nn.functional.mse_loss(
            classification_output, self.label_org)
        #tính loss cho d của phần loss mse của AUs_hat_fake - AUs_target

        # Compute loss with fake images.
        attention_mask, color_regression = self.G(self.x_real, self.label_trg)
        # tính mask attention và color lần đầu
        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.x_real)
        #combined lại để ra hình fake
        critic_output, _ = self.D(x_fake.detach())
        d_loss_critic_fake = torch.mean(critic_output)
        #  tính loss WGAN D_I(fake)


        # Compute loss for gradient penalty.
        alpha = torch.rand(self.x_real.size(0), 1, 1, 1).to(self.device)
        # Half of image info from fake and half from real
        x_hat = (alpha * self.x_real.data + (1 - alpha)
                 * x_fake.data).requires_grad_(True)
        critic_output, _ = self.D(x_hat)
        d_loss_gp = self.gradient_penalty(critic_output, x_hat)
        #Tính loss của d phần gradient penatly

        # Backward and optimize.
        d_loss = d_loss_critic_real + d_loss_critic_fake + self.lambda_cls * \
            d_loss_classification + self.lambda_gp * d_loss_gp
        #Hàm loss tổng của Discriminator  = -gốc + giả + lamda_gp*loss_gp + conditional expression loss

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # Logging.
        self.loss_visualization['D/loss'] = d_loss.item()
        self.loss_visualization['D/loss_real'] = d_loss_critic_real.item()
        self.loss_visualization['D/loss_fake'] = d_loss_critic_fake.item()
        self.loss_visualization['D/loss_cls'] = self.lambda_cls * \
            d_loss_classification.item()
        self.loss_visualization['D/loss_gp'] = self.lambda_gp * \
            d_loss_gp.item()

    def train_generator(self):
        #Tại scale đầu tiên thì input của G là tấm ảnh thật, nhưng khi lên scale thứ hai thì input của G là ảnh tái cấu trúc từ scale trước nó.
        if (self.num_scale <1):
            self.inputG = self.x_real
        else:
            self.inputG = self.imsize(self.x_rec,scale_factor=7/4)


        # Original-to-target domain.
        attention_mask, color_regression = self.G(self.inputG, self.label_trg)
        # mask color và attention được tạo ra từ generator (input là x_real, label_trg)
        # upspamling -> x_real và label_trg

        x_fake = self.imFromAttReg(
            attention_mask, color_regression, self.inputG)
        #x_fake gọi hàm combine 3 thành phần trên

        critic_output, classification_output = self.D(x_fake)
        # đưa hình giả qua Discriminator  (critic_output là WGAN-GP, classification_output là y_hat - AUs)
        g_loss_fake = -torch.mean(critic_output)
        # hàm loss g cho ảnh giả là -
        g_loss_cls = torch.nn.functional.mse_loss(
            classification_output, self.label_trg)
        # hàm loss mse của AUs_hat_fake - AUs_target

        # Target-to-original domain.
        if not self.use_virtual:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            #mask attention và color của ảnh tái tạo là kết quả của x_fake và AUs_gốc
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)
            #tấm ảnh tái tạo combine lại như trên

        else:
            reconstructed_attention_mask, reconstructed_color_regression = self.G(
                x_fake, self.label_org)
            x_rec = self.imFromAttReg(
                reconstructed_attention_mask, reconstructed_color_regression, x_fake)

            reconstructed_attention_mask_2, reconstructed_color_regression_2 = self.G(
                x_fake, self.label_trg_virtual)
            x_fake_virtual = self.imFromAttReg(
                reconstructed_attention_mask_2, reconstructed_color_regression_2, x_fake)

            reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression = self.G(
                x_fake_virtual, self.label_trg)
            x_rec_virtual = self.imFromAttReg(
                reconstructed_virtual_attention_mask, reconstructed_virtual_color_regression, x_fake_virtual.detach())

        # Compute losses
        g_loss_saturation_1 = attention_mask.mean()
        # hàm lỗi chuẩn 2 của mặt nạ attention
        g_loss_smooth1 = self.smooth_loss(attention_mask)
        # hàm lỗi cho mặt nạ đucợ smooth

        if not self.use_virtual:
            g_loss_rec = torch.nn.functional.l1_loss(self.inputG, x_rec)
            #hàm lổi cycle loss nhằm ensure cùng một người ( hình thật và hình tái cấu trúc, chuẩn 1)
            g_loss_saturation_2 = reconstructed_attention_mask.mean()
            # hàm lỗi cho mặt nạ tái tại  L_A
            g_loss_smooth2 = self.smooth_loss(reconstructed_attention_mask)
            # hàm lỗi cho mặt nạ mượt mà hơn

        else:
            g_loss_rec = (1-self.alpha_rec)*torch.nn.functional.l1_loss(self.inputG, x_rec) + \
                self.alpha_rec * \
                torch.nn.functional.l1_loss(x_fake, x_rec_virtual)

            g_loss_saturation_2 = (1-self.alpha_rec) * reconstructed_attention_mask.mean() + \
                self.alpha_rec * reconstructed_virtual_attention_mask.mean()

            g_loss_smooth2 = (1-self.alpha_rec) * self.smooth_loss(reconstructed_virtual_attention_mask) + \
                self.alpha_rec * self.smooth_loss(reconstructed_attention_mask)

        g_attention_loss = self.lambda_smooth * g_loss_smooth1 + self.lambda_smooth * g_loss_smooth2 \
            + self.lambda_sat * g_loss_saturation_1 + self.lambda_sat * g_loss_saturation_2
        #Hàm lỗi attention tổng thể =  lambda_TV * ( loss mượt ảnh gốc + loss mượt ảnh tái cấu trúc) + lanbda_sat *( chuẩn 2 của hai mặt nạ)

        g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + \
            self.lambda_cls * g_loss_cls + g_attention_loss
        #hàm loss tổng = note trong bài báo

        self.reset_grad()
        # reset gradient về 0
        g_loss.backward() #backwark cho g
        self.g_optimizer.step() # tối ưu hóa

        # Logging.
        self.loss_visualization['G/loss'] = g_loss.item()
        self.loss_visualization['G/loss_fake'] = g_loss_fake.item()
        self.loss_visualization['G/loss_rec'] = self.lambda_rec * \
            g_loss_rec.item()
        self.loss_visualization['G/loss_cls'] = self.lambda_cls * \
            g_loss_cls.item()
        self.loss_visualization['G/attention_loss'] = g_attention_loss.item()
        self.loss_visualization['G/loss_smooth1'] = self.lambda_smooth * \
            g_loss_smooth1.item()
        self.loss_visualization['G/loss_smooth2'] = self.lambda_smooth * \
            g_loss_smooth2.item()
        self.loss_visualization['G/loss_sat1'] = self.lambda_sat * \
            g_loss_saturation_1.item()
        self.loss_visualization['G/loss_sat2'] = self.lambda_sat * \
            g_loss_saturation_2.item()
        self.loss_visualization['G/alpha'] = self.alpha_rec

        self.x_rec = x_rec

        if not self.use_virtual:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
            }

        else:
            return {
                "color_regression": color_regression,
                "x_fake": x_fake,
                "attention_mask": attention_mask,
                "x_rec": x_rec,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_attention_mask": reconstructed_attention_mask,
                "reconstructed_color_regression": reconstructed_color_regression,
                "reconstructed_virtual_attention_mask": reconstructed_virtual_attention_mask,
                "reconstructed_virtual_color_regression": reconstructed_virtual_color_regression,
                "x_rec_virtual": x_rec_virtual,
            }



    def print_generations(self, generator_outputs_dict):
        print_epoch_images = False
        save_image(self.denorm(self.x_real), self.sample_dir +
                   '/{}_4real_.png'.format(self.epoch))
        save_image((generator_outputs_dict["color_regression"]+1)/2,
                   self.sample_dir + '/{}_2reg_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_fake"]), self.sample_dir + '/{}_3res_.png'.format(self.epoch))
        save_image(generator_outputs_dict["attention_mask"],
                   self.sample_dir + '/{}_1attention_.png'.format(self.epoch))
        save_image(self.denorm(
            generator_outputs_dict["x_rec"]), self.sample_dir + '/{}_5rec_.png'.format(self.epoch))

        if not self.use_virtual:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

        else:
            save_image(generator_outputs_dict["reconstructed_attention_mask"],
                       self.sample_dir + '/{}_6rec_attention_.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["reconstructed_color_regression"]), self.sample_dir + '/{}_7rec_reg.png'.format(self.epoch))

            save_image(generator_outputs_dict["reconstructed_virtual_attention_mask"],
                       self.sample_dir + '/{}_8rec_virtual_attention.png'.format(self.epoch))
            save_image(self.denorm(generator_outputs_dict["reconstructed_virtual_color_regression"]),
                       self.sample_dir + '/{}_91rec_virtual_reg.png'.format(self.epoch))
            save_image(self.denorm(
                generator_outputs_dict["x_rec_virtual"]), self.sample_dir + '/{}_92rec_epoch_.png'.format(self.epoch))

    def update_tensorboard(self):
        # Print out training information.
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}],  [{}/{}], Epoch [{}/{}]".format(
            et, self.iteration+1, len(self.data_loader), self.epoch+1, self.num_epochs)
        for tag, value in self.loss_visualization.items():
            log += ", {}: {:.4f}".format(tag, value)
        print(log)

        if self.use_tensorboard:
            for tag, value in self.loss_visualization.items():
                self.writer.add_scalar(
                    tag, value, global_step=self.global_counter)



    def animation(self, mode='animate_image'):
        from PIL import Image
        from torchvision import transforms as T

        regular_image_transform = []
        regular_image_transform.append(T.ToTensor())
        regular_image_transform.append(T.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        regular_image_transform = T.Compose(regular_image_transform)

        G_path = sorted(glob.glob(os.path.join(
            self.animation_models_dir, '*G.ckpt')), key=self.numericalSort)[0]
        self.G.load_state_dict(torch.load(G_path, map_location=f'cuda:{self.gpu_id}'))
        self.G = self.G.cuda(0)

        reference_expression_images = []

        with torch.no_grad():
            with open(self.animation_attributes_path, 'r') as txt_file:
                csv_lines = txt_file.readlines()

                targets = torch.zeros(len(csv_lines), self.c_dim)
                input_images = torch.zeros(len(csv_lines), 3, 128, 128)

                for idx, line in enumerate(csv_lines):
                    splitted_lines = line.split(' ')
                    image_path = os.path.join(
                        self.animation_attribute_images_dir, splitted_lines[0])
                    input_images[idx, :] = regular_image_transform(
                        Image.open(image_path)).cuda()
                    reference_expression_images.append(splitted_lines[0])
                    targets[idx, :] = torch.Tensor(
                        np.array(list(map(lambda x: float(x)/5., splitted_lines[1::]))))

        if mode == 'animate_random_batch':
            animation_batch_size = 7

            self.data_iter = iter(self.data_loader)
            images_to_animate, _ = next(self.data_iter)
            images_to_animate = images_to_animate[0:animation_batch_size].cuda(
            )

            for target_idx in range(targets.size(0)):
                targets_au = targets[target_idx, :].unsqueeze(
                    0).repeat(animation_batch_size, 1).cuda()
                resulting_images_att, resulting_images_reg = self.G(
                    images_to_animate, targets_au)

                resulting_images = self.imFromAttReg(
                    resulting_images_att, resulting_images_reg, images_to_animate).cuda()

                save_images = - \
                    torch.ones((animation_batch_size + 1)
                               * 2, 3, 128, 128).cuda()

                save_images[1:animation_batch_size+1] = images_to_animate
                save_images[animation_batch_size+1] = input_images[target_idx]
                save_images[animation_batch_size +
                            2:(animation_batch_size + 1)*2] = resulting_images

                save_image((save_images+1)/2, os.path.join(self.animation_results_dir,
                                                           reference_expression_images[target_idx]))

        if mode == 'animate_image':

            images_to_animate_path = glob.glob(
                self.animation_images_dir + '/*')

            for image_path in images_to_animate_path:
                image_to_animate = regular_image_transform(
                    Image.open(image_path)).unsqueeze(0).cuda()

                for target_idx in range(targets.size(0)):
                    targets_au = targets[target_idx, :].unsqueeze(0).cuda()
                    resulting_images_att, resulting_images_reg = self.G(
                        image_to_animate, targets_au)
                    resulting_image = self.imFromAttReg(
                        resulting_images_att, resulting_images_reg, image_to_animate).cuda()

                    save_image((resulting_image+1)/2, os.path.join(self.animation_results_dir,
                                                                   image_path.split('/')[-1].split('.')[0]
                                                                   + '_' + reference_expression_images[target_idx]))

        # """ Code to modify single Action Units """

        # Set data loader.
        # self.data_loader = self.data_loader

        # with torch.no_grad():
        #     for i, (self.x_real, c_org) in enumerate(self.data_loader):

        #         # Prepare input images and target domain labels.
        #         self.x_real = self.x_real.to(self.device)
        #         c_org = c_org.to(self.device)

        #         # c_trg_list = self.create_labels(self.data_loader)

        #         crit, cl_regression = self.D(self.x_real)
        #         # print(crit)
        #         print("ORIGINAL", c_org[0])
        #         print("REGRESSION", cl_regression[0])

        #         for au in range(17):
        #             alpha = np.linspace(-0.3,0.3,10)
        #             for j, a in enumerate(alpha):
        #                 new_emotion = c_org.clone()
        #                 new_emotion[:,au]=torch.clamp(new_emotion[:,au]+a, 0, 1)
        #                 attention, reg = self.G(self.x_real, new_emotion)
        #                 x_fake = self.imFromAttReg(attention, reg, self.x_real)
        #                 save_image((x_fake+1)/2, os.path.join(self.result_dir, '{}-{}-{}-images.jpg'.format(i,au,j)))

        #         if i >= 3:
        #             break
