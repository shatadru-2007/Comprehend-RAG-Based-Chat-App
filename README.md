# Comprehend-RAG-Based-Chat-App
The user inputs an API key and a text file. The app can then answer the questions the user asks based on the uploaded file.

1. Download the "app.py" file from this repository in a folder.
2. In that folder initiate a virtual environment with Python 3.11.
3. Install the following packages in the virtual environment:
    a. streamlit
    b. google-generativeai
    c. langchain
    d. langchain-google-genai
    e. langchain-community
    f. langchain-core
    g. langchain-classic
    h. faiss-cpu
4. In the terminal use the command "streamlit run app.py" to run the code.
5. Follow the link and open the browser.
6. Visit aistudio.google.com/api-keys and create a Gemini API key and copy it.
7. In the sidebar of Comprehend, paste the API key copied from aistudio.google.com/api-keys and press enter key.
8. Click on "Browse Files" and upload a text (.txt) file.
9. At the bottom of the screen, type the question in the search bar showing "Ask a question..." and press enter key.
10. The query and response will be displayed.
11. Click on "View Sources" to see the portions of the text which were used to generate the response.
