import os


class Config:
    def __init__(self, mode='conv'):
        # project path
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        # image path
        self.image_path = os.path.join(self.project_path, 'image')

        ## Story_Generation_image path
        self.TG_image_path = os.path.join(self.image_path, 'TG.png')
        ## Text_Generation_architecture_image path
        self.TG_architecture_image_path = os.path.join(self.image_path, 'gpt.png')

        ## Summarization_image path
        self.SU_image_path = os.path.join(self.image_path, 'su.jpg')
        ## Summarization_image_architecture_image path
        self.SU_architecture_image_path = os.path.join(self.image_path, 'bart.png')

        ## Poem_Generation_image path
        self.PM_image_path = os.path.join(self.image_path, 'poem.png')

        ## Image_Generation_image path
        self.IG_image_path = os.path.join(self.image_path, 'dalle.png')
        self.IG_dalle_image_path = os.path.join(self.project_path, 'generated')

        # models path
        self.model_path = os.path.join(self.project_path, 'models')
        self.QA_model_path = os.path.join(self.project_path, 'main', 'QA', 'model')
        # document path
        self.docs_path = os.path.join(self.project_path, 'docs')

        # home page
        self.home_title = 'Natural Language Processing, Computer Vision and Creativity'
        self.home_des = "This is a demonstration platform for applying " \
                        "**Natural Language Processing** and **Computer Vision** to real life applications." \
                        "The purpose of the platform is to share knowledge on how to combining Natural Language Processing and computer vision to " \
                        "creativity domain.\n\r" \
                        "This platform provides \n\r" \
                        "* **Question and Answering** for answering the question based on the given context.\n\r" \
                        "* **Story Generation** for auto-generation a story.\n\r" \
                        "* **Story Summarization** for extracting the core meaning of a story."

        # Story Generation page
        self.TG_title = "Make a story..."
        self.TG_des = "One of the cool functions of this demo is to complete your story. The function is designed based " \
                      "on the **OpenAI GPT-2** model which is a causal **transformer** " \
                      "pre-trained language modelling on 40GB of text model. You can type a snippet in the following " \
                      "box and see what happens :)"
        self.TG_model_overview = "The OpenAI GPT-2 language model is a direct successor to GPT (**Generative " \
                                 "Pre-training Transformer**). It expands the unsupervised language model to a much " \
                                 "larger scale by training on a giant collection of free text corpora GPT-2 has **1.5B** " \
                                 "parameters, **10x** more than the original GPT, and it achieves SOTA results on 7 out " \
                                 "of 8 tested language modeling datasets in a zero-shot transfer setting without any " \
                                 "task-specific fine-tuning. The pre-training data contains 8 million Web pages " \
                                 "collected by crawling qualified outbound links from Reddit. Large improvements by " \
                                 "OpenAI GPT-2 are specially noticeable on small datasets and datasets used for " \
                                 "measuring long-term dependency."
        self.TG_about = "OpenAI GPT-2 model was proposed in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by " \
                        "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever."

        # Summarization page
        self.SU_title = "Story Summarization - Extract key information from the given text"
        self.SU_des = "Summarization is the process of summarizing the information in large texts for quicker consumption." \
                      "In our work, we leverage the **BART** model. It is an encoder-decoder model. It converts all language problems " \
                      "into a text-to-text format. You can type a snippet in the following " \
                      "box and see how it helps to generate summarization."
        self.SU_model_overview = "According to the [paper](https://arxiv.org/abs/1910.13461), BART, a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) " \
                                 "corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original " \
                                 "text. It uses a standard Tranformer-based neural machine translation architecture which, despite its simplicity, " \
                                 "can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), " \
                                 "and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best " \
                                 "performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, " \
                                 "where spans of text are replaced with a single mask token. BART is particularly effective when fine tuned for " \
                                 "text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable " \
                                 "training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question " \
                                 "answering, and summarization tasks, with gains of up to 6 ROUGE."
        self.SU_about = "BART model was proposed in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, " \
                        "Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, " \
                        "Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019."


        # Poem page
        self.PM_title = "Automatic Poetry Generator"
        self.PM_des = "Have you thought about using AI to generate a poem? In this demonstration, we design a neural network model, **Bidirectional-LSTM**, which is trained " \
                      "on William Shakespeare’s Sonnets. Simply enter a few words and our computer will generate the remainder of the poem for you. \n\r" \
                      "**Sonnets** \n\r" \
                      "From fairest creatures we desire increase, That thereby beauty's rose might never die,\n\r" \
                      "But, as the riper should by time decease, His tender heir might bear his memory..." \

        # image page
        self.IG_title = "Turn Natural Language into Art."
        self.IG_des = "This demonstration shows how to use SOTA [DALL·E 2](https://openai.com/dall-e-2/), a new AI system, to create realistic images and art from a description in natural language." \
                    "DALL·E 2 can create original, realistic images and art from a description. It can combine concepts, attributes, and styles." \
                    "We use API from [craiyon](https://www.craiyon.com/) to generate images from description."

        # sidebar
        self.nav_menu = ['Home', 'Natural Language Processing', 'Computer Vision']
        self.NLP_menu = ['Make a story', 'Can you summarize', 'Poetry generation', 'Turn language into Art']
        self.CV_menu = ['DeepFake', 'Other']
        self.contr_info = "This an open source project and you are very welcome to contribute your awesome comments, " \
                          "questions and pull requests to the source code (TBC). "
        self.abt_info = 'This app is maintained by Chen Li. You can learn more about me at [VBN](https://vbn.aau.dk/en/persons/142294) of Aalborg University. '

        # NER
        self.ner_model_path = self.project_path + '/spacy_NER/models'
        self.ner_dataset_path = self.project_path + '/spacy_NER/data/Odoo.json'

        # Intents
        self.intent_model_path = self.project_path + '/model/cnn_model.json'
        self.intent_weight_path = self.project_path + '/model/cnn_weights.h5'
        self.intent_RNN_model_path = self.project_path + '/model/rnn_model.json'
        self.intent_RNN_weight_path = self.project_path + '/model/rnn_weights.h5'
        self.intent_LSTM_model_path = self.project_path + '/model/lstm_model.json'
        self.intent_LSTM_weight_path = self.project_path + '/model/lstm_weights.h5'
        self.intent_BERT_model_path = self.project_path + '/model/bert_model.json'
        self.intent_BERT_weight_path = self.project_path + '/model/bert_weights.h5'

        self.bert_model_name = "wwm_uncased_L-24_H-1024_A-16"
        self.bert_ckpt_dir = self.project_path + '/model/' + self.bert_model_name
        self.bert_ckpt_file = self.project_path + '/model/' + self.bert_model_name + '/bert_model.ckpt'
        self.bert_config_file = self.project_path + '/model/' + self.bert_model_name + '/bert_config.json'
        self.vocab_file = self.project_path + '/model/' + self.bert_model_name + '/vocab.txt'
        self.log_dir = self.project_path + '/model/log/intent_detection/'

        # googlenewsvector
        self.google_news_path = 'C:/Users/Admin/Downloads/googlenews-vectors-negative300.bin.gz'

        # cnn/rnn parameters
        self.maxlen = 400
        self.steps_per_epoch = 10
        self.validation_steps = 4
        self.embedding_dims = 300
        self.filters = 250
        self.kernel_size = 3
        self.hidden_dims = 250
        self.epochs = 10
        self.num_class = 10
