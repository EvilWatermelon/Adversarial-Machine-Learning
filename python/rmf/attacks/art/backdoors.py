from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor

# Executing the PoisoningAttackCleanLabelBackdoor attack
def clean_label():
    return "test"

# Executing the PoisoningAttackBackdoor
def art_poison_backdoor_attack(perturbation, x, y, broadcast):
    backdoor_class = PoisoningAttackBackdoor(perturbation)
    backdoor_class.poison(x, y, broadcast)
