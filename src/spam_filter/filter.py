import numpy as np
import pandas as pd
from email.parser import Parser
from email import message_from_string
import re



one_file = ('./dataset/easy_ham/0001.ea7e79d3153e7469e7a9c3e0af6a357e')

email_message = message_from_string(open(one_file).read())

# Extract the email body
if email_message.is_multipart():
    # If multipart, iterate through parts
    for part in email_message.walk():
        # Only extract text/plain or text/html content
        if part.get_content_type() == "text/plain":
            email_text = part.get_payload(decode=True).decode(part.get_content_charset())
            break
else:
    # For non-multipart emails
    email_text = email_message.get_payload(decode=True).decode(email_message.get_content_charset())


# Use regular expressions to change the email text




# Save to a text file
file_path = "email_body.txt"
with open(file_path, "w") as file:
    file.write(email_text)



#sample_email = open(one_file, 'r').read()   # read the file
#print(sample_email)
