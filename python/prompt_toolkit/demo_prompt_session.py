from prompt_toolkit import PromptSession


session = PromptSession()

while True:
    text = session.prompt("Well? ")
    print(repr(text))
    if text == 'quit':
        break
