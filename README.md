# NotebookToNotepad

Keyboarding remains the most common way of inputting data into computers. This is a time consuming and labor intensive operation. 

Notebook To Notepad would convert handwritten text in images to a notepad file with the same data, thereby saving the user both time and energy.

### How to run:
1. Unzip NTNBackEnd\model\Snapshots.zip and move snapshot-5 to NTNBackEnd\model 
2. Run command "python manage.py runserver"

### Output:
1. Output is displayed on the webpage 'result'
2. In addition to that, output.txt file is created which can be returned to the user.

### Future Scope:
1. Add tf.reset_default_graph() to avoid having to run server after every input.
2. Improve accuracy of model.
3. Change dataset and train in a manner which allows input of images with multiple lines/paragraphs (Apply Segmentation).
