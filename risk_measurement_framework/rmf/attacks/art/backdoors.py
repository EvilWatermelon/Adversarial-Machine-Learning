from art.attacks.poisoning import *
from art import *
from art.attacks.poisoning.perturbations import insert_image, add_pattern_bd

import matplotlib.pyplot as plt
import numpy as np

def poison_func(x):
    return insert_image(x, backdoor_path='../rmf/backdoors/alert.png',
                        size=(10, 10), mode='RGB', blend=0.8, random=True)

def mod(x):

    patch_size = 8
    x_shift = 32 - patch_size - 5
    y_shift = 32 - patch_size - 5

    original_dtype = x.dtype
    x = insert_image(x, backdoor_path='../rmf/backdoors/htbd.png',
                     channels_first=False, random=False, x_shift=x_shift, y_shift=y_shift,
                     size=(patch_size, patch_size), mode='RGB', blend=1)
    return x.astype(original_dtype)

# Executing the PoisoningAttackCleanLabelBackdoor attack (black-box)
def clean_label(x, y, clf, target_label):
    """
    https://people.csail.mit.edu/madry/lab/cleanlabel.pdf
    """

    backdoor = PoisoningAttackBackdoor(add_pattern_bd)
    attack = PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, proxy_classifier=clf,
                                               target=target_label, pp_poison=.33, norm=2, eps=5,
                                               eps_step=0.1, max_iter=200)

    x.reshape(-1, 30, 30, 3)
    print(x.shape)

    poison_data, poison_labels = attack.poison(x, y)
    return poison_data, poison_labels

# Untargeted attack (black-box)
def art_poison_backdoor_attack(x, y, num_of_images):
    """
    https://arxiv.org/abs/1708.06733
    """

    n_train = np.shape(x)[0]
    num_selection = num_of_images

    random_selection = np.random.choice(n_train, num_selection)

    temp_x = x[random_selection]

    del_x = np.delete(x, [random_selection], axis=0)

    y = y[random_selection]

    backdoor_class = PoisoningAttackBackdoor(poison_func)
    poisoned_x, poisoned_y = backdoor_class.poison(temp_x, y)

    poisoned_data = np.concatenate((del_x, poisoned_x), axis=0)

    print("Finished poisoning!")

    return poisoned_data, poisoned_y

# Targeted attack (white-box)
def art_hidden_trigger_backdoor(x, y, target, source):
    """
    https://arxiv.org/pdf/1910.00033.pdf
    """
    backdoor = PoisoningAttackBackdoor(mod)

    poison_attack = HiddenTriggerBackdoor(classifier, eps=16/255, target=target, source=source, feature_layer=9, backdoor=backdoor,
                                          learning_rate=0.01, decay_coeff = .1, decay_iter = 1000, max_iter=3000, batch_size=25, poison_percent=.15)

    poison_data, poison_labels = poison_attack.poison(x, y)

    """
    # Create finetuning dataset
    dataset_size = size
    num_labels = label_size
    num_per_label = dataset_size/num_labels

    poison_dataset_inds = []

    for i in range(num_labels):
      label_inds = np.where(np.argmax(y_train, axis=1) == i)[0]
      num_select = int(num_per_label)
      if np.argmax(target) == i:
          num_select = int(num_select - min(num_per_label, len(poison_data)))
          poison_dataset_inds.append(poison_indices)

      if num_select != 0:
          poison_dataset_inds.append(np.random.choice(label_inds, num_select, replace=False))

    poison_dataset_inds = np.concatenate(poison_dataset_inds)

    poison_x = np.copy(x_train)
    poison_x[poison_indices] = poison_data
    poison_x = poison_x[poison_dataset_inds]

    poison_y = np.copy(y_train)[poison_dataset_inds]

    """

    return poison_data, poison_labels
