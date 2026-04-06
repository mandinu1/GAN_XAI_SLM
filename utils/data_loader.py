import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

def get_data_loader(args, feature_extraction=False, target_class=None):
    
    if feature_extraction:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5))
        ])

    train_dataset = datasets.ImageFolder(
        root=f"{args.dataroot}/train",
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{args.dataroot}/val",
        transform=transform
    )

    # Filter to only one class if specified
    if target_class is not None:
        if target_class not in train_dataset.class_to_idx:
            raise ValueError(f"Class '{target_class}' not found in dataset.")

        class_idx = train_dataset.class_to_idx[target_class]

        train_dataset.samples = [
            s for s in train_dataset.samples if s[1] == class_idx
        ]
        train_dataset.imgs = train_dataset.samples
        train_dataset.targets = [s[1] for s in train_dataset.samples]

        test_dataset.samples = [
            s for s in test_dataset.samples if s[1] == class_idx
        ]
        test_dataset.imgs = test_dataset.samples
        test_dataset.targets = [s[1] for s in test_dataset.samples]

        print(f"Using only class: {target_class}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    train_loader = data_utils.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
    )
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    return train_loader, test_loader