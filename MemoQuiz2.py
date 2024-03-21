# pip install langchain openai chromadb pypdf pip install PyMuPDF python-dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

from nicegui import events, ui
import os

richtige_quiz_antwort=""
topic_list=[]
topic_picker= None

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# RAG prompt
from langchain.prompts import PromptTemplate

prompt_template = """You are a German quiz master and you create one multiple choice question and answers for a quiz.
                Create one single, unique detailed question about a {question}.
                The answers should be numbered, unique and for experts.
                Use exclusively only the context to create a multiple choice question about a topic.                
                Provide the right answer at the end of the output.
                Always speak German.


                Detailed Question: {question}

                  \n\n1: 

                  \n\n2:

                  \n\n3:

                  \n\n4: 

                Correct Answer:

                {context}

"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


from langchain_openai import OpenAIEmbeddings

vectorstore2 = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())

# LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

# RetrievalQA
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore2.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt}
)



# Funktion um mit einer LLMChain Themenvorschläge zu generieren
async def get_topics():
    if text_input_password.value == "0815":
        label_quiz_themen = ui.label().style('white-space: pre-wrap;')

        print("### inside get_topics")

        prompt_template2 = """Create a list of 20 contect related terms for a quiz about {question}. 
        Only list terms that are mentioned in the {context}.
        Don't create a numbered list.
        Don't invent any terms.
        Always speak German.
        
        List of terms about {question}:
        first term, second term, third term
        
        """

        prompt2 = PromptTemplate(
            template=prompt_template2, input_variables=["context", "question"]
        )

        qa_chain2 = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore2.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt2}
        )

        # Assuming qa_chain is now an async function or you're using an async equivalent
        topics= await qa_chain2.arun({"query": text_input_user_question.value})
        label_quiz_themen.set_text(topics)
        topic_list = topics.split(", ")
        print (topic_list)
        # Create the dropdown menu and attach the event handler
        # global topic_picker = ui.select(options=topic_list, label='Choose an option', on_change=on_selection_change)
        for item in topic_list:
            topic_picker.options.append(item)
        topic_picker.update()
# Event handler for when the selection changes
async def on_selection_change(event):
    print ("### inside on_selection_change")
    selected_option = topic_picker.value  # Access the selected item
    ui.notify(f'Frage zu: {selected_option} wird erstellt.')
    text_input_user_question.set_value(selected_option)
    await create_question()


# Funktion namens backup_db zum Backup der des Verzeichnisses db und aller darin enthaltenen Dateien mit Python als ZIP-Datei mit Zeitstempel
def backup_db():
   import zipfile
   from datetime import datetime
   timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
   # Pfad zum Ordner mit den Daten
   data_folder = 'db'
   zipfile_name = 'db_' + timestamp + '.zip'
   with zipfile.ZipFile(zipfile_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
       for root, dirs, files in os.walk(data_folder):
           for file in files:
               zipf.write(os.path.join(root, file))

# Funktion namens delete_db zum Löschen der des Verzeichnisses db und aller darin enthaltenen Dateien mit Python
def delete_db():
   # Pfad zum Ordner mit den Daten
   data_folder = 'db'
   backup_db()
   import shutil
   shutil.rmtree(data_folder)

def vector_upload(text):

    print ("### inside vector_upload")

    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    from langchain_community.document_loaders import TextLoader

    text_loader = TextLoader(text)
    data = text_loader.load()
    print ("### data #############################")
    # print (data)


    # Split
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100,separators="\n\n")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500)
    all_splits = text_splitter.split_documents(data)

    # Store splits
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="db")
    vectorstore.persist()


from nicegui import ui, events

def refresh_page():
    ui.run_javascript('window.location.reload()')

def handle_pdf_upload(event: events.UploadEventArguments):
    print ("### inside handle_pdf_upload")
    with event.content as f:
        file = open('pdffile.pdf', 'wb')
        for line in f.readlines():
            file.write(line)
        file.close()
    load_pdf_document('pdffile.pdf')

def load_pdf_document(file):
    print ("### inside load_pdf_document")
    text=""
    from langchain.document_loaders import PyMuPDFLoader
    print (f'loading {file}')
    loader = PyMuPDFLoader(file)
    data = loader.load()
    for item in data:
        text = text + (item.page_content)
    # write file to disk
    with open('textfile.txt', 'w') as f:
        f.write(text)
    handle_upload()

def handle_upload():
    print ("### inside handle_upload")
    import os
    os.environ["OPENAI_API_KEY"] = "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"

    embeddings = OpenAIEmbeddings()

    from langchain.document_loaders import TextLoader
    loader = TextLoader("textfile.txt")
    docs = loader.load()

    # split up the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(docs)

    # create the db in a folder called db
    persist_directory = 'db'

    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vectorstore.persist()

async def create_question():
    if text_input_password.value == "0815":
        global richtige_quiz_antwort
        print("### inside create_question")
        # Assuming qa_chain is now an async function or you're using an async equivalent
        first_result = await qa_chain.arun({"query": text_input_user_question.value})
        #print (first_result)
        #print (type(first_result))
        parts = first_result.rsplit('\n\n', 1)
        label_quiz_frage.set_text(parts[0])
        # label_quiz_antwort.set_text(parts[1])
        print("Frage: " +parts[0])
        print("Antwort: " +parts[1])
        richtige_quiz_antwort = parts[1]

def check_answer():
    label_quiz_antwort.set_text(richtige_quiz_antwort)


def write_inputs():
    create_question.refresh()

def clear_inputs():
    print ("### inside clear_inputs")
    text_input_user_question.value = ''
    text_input_user_anwer.value = ('')
    label_quiz_frage.set_text('')
    label_quiz_antwort.set_text('')


with ui.row():
    with ui.column():
        label_quiz_frage = ui.label().style('white-space: pre-wrap;')
        #markdown_area_quiz_frage = ui.markdown('').style('color: black; font-size: 20px;width: 800px;height: 800px;')
        label_quiz_antwort = ui.label('').style('color: blue; font-size: 16px;width: 800px;')



    with ui.column():
        topic_picker = ui.select(options=topic_list, label='Wähle ein Thema', on_change=on_selection_change)
        text_input_user_question = ui.input('Frage zu welchem Thema?').style('width: 300px;')
        button_frage_senden = ui.button('Frage erstellen', on_click=create_question)
        text_input_user_anwer = ui.input('Deine Antwort:').style('width: 300px;')
        button_antwort_senden = ui.button('Antwort senden', on_click=check_answer)
        button_eingaben_loeschen = ui.button('Anzeige löschen', on_click=clear_inputs)
        with ui.expansion('Erweiterte Funktionen!', icon='work').classes('w-full'):
            #ui.upload(on_upload=handle_upload).props('accept=.txt').classes('max-w-full')
            ui.upload(on_upload=handle_pdf_upload).props('accept=.pdf').classes('max-w-full')
            button_datenbank_loeschen = ui.button('Datenbank löschen', on_click=delete_db)
            text_input_password = ui.input('Passwort:').style('width: 300px;')
            button_themengebiete = ui.button('Themengebiete', on_click=get_topics)


ui.run()
