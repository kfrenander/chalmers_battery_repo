def prompt_user_msmt_setting():
    print("Please choose one of the following options:")
    print("1. Option 1 (fast)")
    print("2. Option 2 (accurate)")
    print("3. Option 3 (default)")
    choice = input("Enter your choice (fast, accurate, or default): ").lower()
    return choice


def prompt_user_device_setting():
    print("Please choose one of the following options:")
    print("1. Option 1 (combined)")
    print("2. Option 2 (individual)")
    print("3. Option 3 (default)")
    choice = input("Enter your choice (combined, individual, or default): ").lower()
    return choice


def choice_handler_msmt_setting():
    options = ["fast", "accurate", "default"]
    choice = prompt_user_msmt_setting()

    if choice in options:
        print("You chose", choice)
    elif choice == "1":
        choice = "fast"
        print("You chose Option 1 (fast)")
    elif choice == "2":
        choice = "accurate"
        print("You chose Option 2 (accurate)")
    elif choice == "3":
        choice = "default"
        print("You chose Option 3 (default)")
    else:
        print("Invalid choice. Setting default value.")
        choice = "default"  # Setting default value
        print("Default option chosen: default")
    return choice


def choice_handler_device_setting():
    options = ["combined", "individual", "default"]
    choice = prompt_user_device_setting()

    if choice in options:
        print("You chose", choice)
    elif choice == "1":
        choice = "combined"
        print("You chose Option 1 (combined)")
    elif choice == "2":
        choice = "individual"
        print("You chose Option 2 (individual)")
    elif choice == "3":
        choice = "default"
        print("You chose Option 3 (default / combined)")
    else:
        print("Invalid choice. Setting default value.")
        choice = "default"  # Setting default value
        print("Default option chosen: default")
    return choice


def main():
    my_accuracy_setting = choice_handler_msmt_setting()
    print(f'Setting has been set to \'{my_accuracy_setting}\'')
    my_device_setting = choice_handler_device_setting()
    print(f'Setting has been set to \'{my_device_setting}\'')


if __name__ == "__main__":
    main()
