# This is a sample Python script.

from utils.file_manager import FileManager

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filenames, train_set, test_set, ground_truths = FileManager.prepare_data(base_path='data/ears/')


