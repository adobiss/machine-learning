import math
from PIL import Image

myImage = Image.open("D:\ML\Datasets\dec_tree_dataset_example.png")
myImage.show()

# In a set of binary classes (one or the other) of YES or NO,
# 9/14 are YES and 5/14 are NO ('Play Tennis' feature).

S = 14
yes = 9
no = 5
sunny = 5
rain = 5
overcast = 4

p_yes = yes / S
p_no = no / S
p_sunny = sunny / S
p_rain = rain / S
p_overcast = overcast / S
 

# Calculate the entropy of the dataset S:
H_complete = -p_yes * math.log(p_yes, 2) - p_no * math.log(p_no, 2)

H_outlook_sunny = -(2/5) * math.log(2/5, 2) - (3/5) * math.log(3/5, 2)
H_outlook_rain = -(3/5) * math.log(3/5, 2) - (2/5) * math.log(2/5, 2)
H_outlook_overcast = -(4/4) * math.log(4/4, 2) - 0

# Average Entropy for Outlook

I_Outlook = p_sunny * H_outlook_sunny + p_rain * H_outlook_rain
+ p_overcast * H_outlook_overcast

# Information gain

Information_gain = H_complete - I_Outlook

print(round(Information_gain, 3))