from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import insert_image

def poison_func(x):
    return insert_image(x, backdoor_path='alert.png',
                        size=(10,10), mode='RGB', blend=0.8, random=True)

# Executing the PoisoningAttackCleanLabelBackdoor attack
def clean_label(pattern):
    """
    https://people.csail.mit.edu/madry/lab/cleanlabel.pdf
    """
    backdoor = PoisoningAttackBackdoor(pattern)
    attack = PoisoningAttackCleanLabelBackdoor(backdoor=backdoor, proxy_classifier=proxy.get_classifier(),
                                           target=targets, pp_poison=percent_poison, norm=2, eps=5,
                                           eps_step=0.1, max_iter=200)

# Executing the PoisoningAttackBackdoor
def art_poison_backdoor_attack(x, y, broadcast):
    """
    https://arxiv.org/abs/1708.06733
    """
    backdoor_class = PoisoningAttackBackdoor(poison_func)
    backdoor_class.poison(x, y, broadcast)
