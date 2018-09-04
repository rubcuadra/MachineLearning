import numpy as np
from glob import glob
from collections import Counter
from random import shuffle
from enum import Enum
'''
The dataset is already divided into 10 parts for cross-validation. 
The task is to classify spam. 
Spam messages contain spmsg in their title, normal messages contain legit.
The text of the letter itself consists of two parts: the subject and the body of the letter. 
All words are replaced by int corresponding to their index in some global dictionary (a kind of anonymization). 
Accordingly, you are required to build a naive Bayesian classifier and, in doing so,

1) Come up with, or test what you can do with the subject and body's letter to improve the quality of work.
2) How to take into account (or not to take into account) the words that may occur in the training sample, but may not meet in the test sample and vice versa.
3) How to impose additional restrictions on your classifier so that good letters almost never get into spam, but at the same time, perhaps the overall quality of the classification hasn't decreased too much.
4) Understand how the classifier is arranged inside and be able to answer any questions about the theory associated with it.

Based on http://www.cs.ubbcluj.ro/~gabis/DocDiplome/Bayesian/000539771r.pdf
'''
class EmailType(Enum):
    SPAM = 0
    LEGIT = 1

class Email(object):
    """Subject and Body are Counter objects Ej. Counter({432:3,121:1,2321:0})"""
    def __init__(self,subject,body,etype=EmailType.LEGIT,name=""):
        super(Email, self).__init__()
        self.subject = subject
        self.body = body
        self.type = etype 
        self.words = body+subject
        self.name = name

    @staticmethod
    def parseFromFile(pathToFile):
        t = EmailType.SPAM if "spmsg" in pathToFile else EmailType.LEGIT 
        #Split in subject and body
        with open(pathToFile,"r") as f:
            subject = next(f).strip("Subject: ") #Add Subject
            next(f) #Skip the division between subject and body
            body = next(f) #Get Body
            #Create the Count Object
            subC = Counter( int(k) for k in subject.split() )
            bodC = Counter( int(k) for k in body.split() )
        return Email(subC,bodC,etype=t,name=pathToFile)

    def __str__(self):  return str(self.name)
    def __repr__(self): return str(self.name)

    def __iter__(self):
        return ( word for word in self.words )
    def __getitem__(self, key):
        return self.words[key]

class EmailNaiveBayesClassifier(object):
    """docstring for EmailNaiveBayesClassifier"""
    def __init__(self, trainingEmails):
        super(EmailNaiveBayesClassifier, self).__init__()
        self.numSpamEmails, self.numLegitEmails = 0, 0
        #Split into classes
        spam_subjects,  spam_bodies  = Counter(),Counter()
        legit_subjects, legit_bodies = Counter(),Counter()
        for email in trainingEmails:
            if email.type is EmailType.SPAM:
                spam_subjects += email.subject 
                spam_bodies   += email.body
                self.numSpamEmails  += 1  
            else:
                legit_subjects += email.subject 
                legit_bodies   += email.body 
                self.numLegitEmails += 1
        
        #Initial probabilities
        self.P_legit = self.numLegitEmails/len(trainingEmails) #P(Legit)
        self.P_spam = self.numSpamEmails/len(trainingEmails)   #P(Spam)
        self.totalEmails = self.numLegitEmails+self.numSpamEmails

        #SpamSubjects, SpamBodies
        self.ss, self.sb = spam_subjects,  spam_bodies
        #LegitSubject, LegitBody
        self.ls, self.lb = legit_subjects, legit_bodies
        #All words
        self.spamWords = self.ss+self.sb
        self.legitWords = self.ls+self.lb

    #Spam Example
    #word => Word that we want to estimate probability
    #setOfWords => Counter Dictionary {word:numOfTimesItAppears}
    #pc => P(c) ; Probability of an email belonging to that class
    #P(W n S) 
    #P(W intersec S) => TimesTheWordAppearsOn_Spam_Emails/TotalEmails
    def probabWordBelongsTo(self,word,setOfWords):
        return (setOfWords[word]/self.totalEmails) #self.P_legit|self.P_spam 

    #only searches trough words, despite where they appear
    def classify(self, email, threshold=0):
        r = np.log(self.P_spam/self.P_legit)  #Initial value, probab it is spam
        for word in email:
            pws = self.probabWordBelongsTo(word,self.spamWords) #Spam
            pwl = self.probabWordBelongsTo(word,self.legitWords) #Legit
            if pws == 0.0 or pwl ==0.0: continue #We don't know this word, just skip it
            r+= np.log( pws/pwl )
        return EmailType.SPAM if r>threshold else EmailType.LEGIT

    #Match subjects with subjects and body with body, different dictionaries
    def classify2(self, email, threshold=0):
        r = np.log(self.P_spam/self.P_legit)  #Initial value, probab it is spam
        
        for word in email.subject:
            #Check Subjects
            pwss = self.probabWordBelongsTo(word,self.ss)  #Spam
            pwls = self.probabWordBelongsTo(word,self.ls) #Legit
            if not (pwss==0.0 or pwls==0.0): #We don't know this word
                r+= np.log( pwss/pwls )

            #Check Body
            pwsb = self.probabWordBelongsTo(word,self.sb)  #Spam
            pwlb = self.probabWordBelongsTo(word,self.lb) #Legit
            if not (pwsb == 0.0 or pwlb ==0.0): #We don't know this word
                r+= np.log( pwsb/pwlb )

        return EmailType.SPAM if r>threshold else EmailType.LEGIT

def getEmailsFromFolders(pathToFolders,cross_validation_p=0.0):
    #Split Folders
    folders = glob(f'{pathToFolders}/*') ; shuffle(folders) #Get and Shuffle
    splitIx = int(len(folders)*cross_validation_p)          #Index for splitting
    #Prepare Paths and Returns
    forTestFolders, forTrainFolders = folders[:splitIx], folders[splitIx:]
    trainEmails,testEmails = [],[]
    #Fill returns
    for path in forTestFolders:
        for emailPath in glob(f"{path}/*"):
            testEmails.append( Email.parseFromFile(emailPath) )
    for path in forTrainFolders:
        for emailPath in glob(f"{path}/*"):
            trainEmails.append( Email.parseFromFile(emailPath) )
    #Return data
    return trainEmails,testEmails

if __name__ == '__main__':
    trainEmails, testEmails = getEmailsFromFolders('./emails',cross_validation_p=0.1)
    ENBC = EmailNaiveBayesClassifier(trainEmails)    
    classifiers = (ENBC.classify,ENBC.classify2)
    
    for classifier in classifiers:
        print(f"Using: {classifier.__qualname__}") #__name__
        incorrects = 0
        missCSpam,missCLegit = 0,0

        for email in testEmails:
            c = classifier(email, threshold=25)        
            if(c is email.type): continue
            
            if email.type is EmailType.SPAM: missCSpam+=1
            else: missCLegit+=1
            incorrects+=1

        print(f"E: {incorrects}/{len(testEmails)} => {format(100-100*incorrects/len(testEmails),'.2f')}% Accuracy")
        print(f"SPAM sent to LEGIT {missCSpam}")
        print(f"LEGIT sent to SPAM {missCLegit}")

        


