answer_list = list()

q1 = input("Is your training data from a public source? [y/n]: ")

if q1 is 'y' or q1 is 'yes':
    print("Thank you.")
elif q1 is 'n' or q1 is 'no':
    print("Thank you.")
else:
    print("Please answer the question with 'y', 'yes' or 'n', 'no'! ")

answer_list.append(q1)

q2 = input("Is your training data encrypted? [y/n]: ")

if q2 is 'y' or q1 is 'yes':
    print("Thank you.")
elif q2 is 'n' or q1 is 'no':
    print("Thank you.")
else:
    print("Please answer the question with 'y', 'yes' or 'n', 'no'! ")

answer_list.append(q2)
