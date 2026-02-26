from rag.retrieve import retrieve


print("🩺 Healthcare AI Assistant (Privacy Preserved)")
print("Type 'exit' to quit")

while True:
    q = input("\nUser: ")
    if q.lower() == "exit":
        break

    answers = retrieve(q)
    print("\nAssistant:")
    for a in answers:
        print("-", a)
