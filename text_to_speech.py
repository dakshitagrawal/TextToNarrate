import boto3
import os
from contextlib import closing
import sys
import re

class audio_generator(object):
    def __init__(self, ACCESS_ID, ACCESS_KEY, region_name):
        self.ACCESS_ID = ACCESS_ID
        self.ACCESS_KEY = ACCESS_KEY
        self.region_name = region_name

    def get_client(self, ACCESS_ID, ACCESS_KEY, region_name):
        client = boto3.client('polly',
                          aws_access_key_id=ACCESS_ID,
                          aws_secret_access_key=ACCESS_KEY,
                          region_name= region_name)

        return client

    def get_response(self, client, text):

        response = client.synthesize_speech(
            OutputFormat='mp3',
            Text= text,
            VoiceId='Joanna'
        )

        return response

    def split_into_sentences(self, text):
        caps = "([A-Z])"
        digits = "([0-9])"
        prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
        suffixes = "(Inc|Ltd|Jr|Sr|Co)"
        starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
        acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
        websites = "[.](com|net|org|io|gov|me|edu)"

        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = text.replace("—", " ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
        text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        if "..." in text: text = text.replace("...","<prd><prd><prd>")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def paragraph_creator(self, sentences, sentences_length):
        count = 0
        segment = ""

        sentences_segments = []

        for i in range(len(sentences)):
            segment = segment + " " + sentences[i]

            count+=1

            if (count == sentences_length or i == len(sentences)-1):
                count = 0
                sentences_segments.append(segment)
                segment = ""

        return sentences_segments

    def save_audio(self, response, folder_name = "audio", file_name = "audio_file_00"):

        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:

                if (not os.path.isdir(folder_name)):
                    os.mkdir(folder_name)

                try:
                    # Open a file for writing the output as a binary stream
                    with open(folder_name + "/" + file_name, "wb") as file:
                        file.write(stream.read())
                except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    sys.exit(-1)

    def final_audio_files(self, paragraphs, client):
        for i in range(len(paragraphs)):

            response = self.get_response(client, paragraphs[i])
            self.save_audio(response, file_name = ("audio_file_" + str(i+1) + ".mp3"))

    def generate_audio(self, text, no_of_sentences):
        client = self.get_client(self.ACCESS_ID, self.ACCESS_KEY, self.region_name)
        sentences = self.split_into_sentences(text)
        paragraphs = self.paragraph_creator(sentences, no_of_sentences)
        self.final_audio_files(paragraphs, client)
        
        return paragraphs


ACCESS_ID = "AKIAJYRP5HASIB34EUPA"
ACCESS_KEY = "V0ayIj0dULvBbLl9ziH/3Pvb8kXxJ/D3ba2YHHej"
region_name = "us-east-1"

text = ""

audio = audio_generator(ACCESS_ID, ACCESS_KEY, region_name)
paragraphs = audio.generate_audio(text, 3)
print (paragraphs)
