from prompt_toolkit import prompt


while True:
    text = prompt("Well? ")
    print(repr(text))
    if text == 'quit':
        break
