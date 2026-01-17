
# DB-MoRE: Dynamic Bias-Aware Mixture-of-Reconstruction-Experts for Long-Tailed OOD Detection

##Training the whole model using DB-MoRE framework

CIFAR10-LT: 

```
python train.py --gpu 0 --epochs 200 --lr 1e-3 --ds cifar10  \
    --rho 0.01  --md ResNet18  \
	--drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

CIFAR100-LT:

```
python train.py --gpu 0 --epochs 200 --lr 1e-3 --ds cifar10  \
    --rho 0.01  --md ResNet18  \
	--drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

ImageNet-LT:

```
python train.py --gpu 0 --epochs 100 --ds imagenet  \
    --md ResNet50 -e 60 --opt sgd --decay multisteps --lr 0.1 --wd 5e-5 --tb 100 \
    --ddp --dist_url tcp://localhost:23457 \
	--drp <where_you_store_all_your_datasets> --srp <where_to_save_the_ckpt>
```

## Testing

CIFAR10-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar10 --dout $dout \
	--drp <where_you_store_all_your_datasets> \
	--ckpt_path <where_you_save_the_ckpt>
done
```

CIFAR100-LT:

```
for dout in texture svhn cifar tin lsun places365
do
python test.py --gpu 0 --ds cifar100 --dout $dout \
	--drp <where_you_store_all_your_datasets> \
	--ckpt_path <where_you_save_the_ckpt>
done
```

ImageNet-LT:

```
python test.py --gpu 0  \
	--drp <where_you_store_all_your_datasets> \
	--ckpt_path <where_you_save_the_ckpt>
```



