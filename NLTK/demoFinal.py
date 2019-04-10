import nltk
import nltk.corpus
import json
import string
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster import VectorSpaceClusterer, cosine_distance
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from itertools import islice
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np


class Proximity_finder_for_documents:
              
    def __init__(self,corpora,num_of_nearest_neighbors=1,max_term_count=1000,proximity_distance=80.7):
        self.num_of_nearest_neighbors=num_of_nearest_neighbors
        self.max_term_count=max_term_count
        self.corpora=corpora
        self.proximity_distance=proximity_distance
        self.referer={}
        
        index=0
       
        for i in self.corpora.keys():
            self.referer[index]=i
            refined=self.corpora[i].translate(string.punctuation)
            self.corpora[i]=refined
                   
                   
    def stem_tokens(self,tokens):
        stemmed = []
        not_useful_words='And', 'we''but', 'will','would','it','that','for', 'nor', 'or', 'so', 'neither','an', 'a', 'the', 'is', 'was', '.', 'yet'
        for word in tokens: 
           if word in not_useful_words:
               tokens.remove(word)
        stemmer=PorterStemmer()
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return Counter(stemmed).most_common(self.max_term_count)
    
    def word_lemmetizer(self,tokens):
        lemmetized=[]
        lmtzr = WordNetLemmatizer()
        for item in tokens:
            lemmetized.append(lmtzr.lemmatize(item))
        return Counter(lemmetized).most_common(self.max_term_count)
        
    
    def tokenize(self,text):
        return self.stem_tokens(nltk.word_tokenize(text))
        
        #return self.word_lemmetizer(nltk.word_tokenize(text))
        
    def initialize_transform(self,tf_idf):
        self.tf_idf=tf_idf
        
        
    def get_nearest_neighbors(self,proximity):
       result=[]
       for i in range(self.num_of_nearest_neighbors):
           result.append(proximity[i][0])
       return result
    
    def get_similar_document(self,proximity):
        result=[]
        diameter=proximity[len(proximity)-1][1]
        value=(self.proximity_distance*diameter)/100
        for i in range(len(proximity)):
          if(proximity[i][1]<=value):
              result.append(proximity[i][0])
          else:
              break
        return result
              
    
    def fetch_from_corpora(self,indices):
        result=[]
        for i in indices:
            result.append(self.corpora[i])
        return result   
        
    def tf_idf_corpora(self,dict_refined_corpus):
        vector_corpora=self.tf_idf.fit_transform(dict_refined_corpus.values())
        newdict=vector_corpora.toarray()
        return newdict
        
    
    def tf_idf_query(self,query_document):
        vector_query_unequal_dimension=self.tf_idf.transform([query_document.translate(string.punctuation)])
        return vector_query_unequal_dimension.toarray()
         
       
        
    def find_proximity(self,query_document):
        temp={}
        proximity={}
        index=0
        ids=list(self.corpora.keys())
        
        self.initialize_transform(TfidfVectorizer(tokenizer=self.tokenize,stop_words='english',lowercase=False))
        #make a dictionary and give an unique identifier to each document in the corpus
           
        #now dict_corpus contains a dictionary with each document as value and unique number as key (key from 0 to num_of_documents)
        vector_corpora=self.tf_idf_corpora(self.corpora)
        vector_query=self.tf_idf_query(query_document)
        
        number_of_zeros=0;
        for x in np.nditer(vector_query):
            if(x!=0.0):
                break
            else:
                number_of_zeros=number_of_zeros+1
                
                
        if(number_of_zeros>=vector_query.size):
             return " "
             
        for i in ids:
          temp[i]=nltk.cluster.util.cosine_distance(vector_query[0], vector_corpora[index])
          
          #temp[i]=nltk.cluster.util.euclidean_distance(vector_query[0], vector_corpora[index])
          index=index+1
        
        
        proximity = sorted(temp.items(), key=lambda kv: kv[1])
        
        #return self.get_nearest_neighbors(proximity)
        return self.get_similar_document(proximity)
    
       
def main():
    corpora={}
    corporaEncoded="246904!!!  This risk register item will be used to track risk analysis & remediation activities associated with HIPAA compliance activities. ^^^246905!!!  An annual security audit by a 3rd party consultancy identified that senstive customer information is flowing unencrypted over the organization's internal network. ^^^246906!!!  Applications, systems or platforms do not have the capability to enforce access rules on users to limit access to data based upon user role, identity or privileges. ^^^246907!!!  Customer run-off and the inability to attract new customers at current rates ^^^246908!!!The organization does not have the capability to manage accounts giving access to internal systems leading to poor data protection, lack of non-repudiation or accountability.^^^246909!!!  Losses due to the existence and management of accounts receivable ^^^246910!!!  Losses due to the default of Broker-Dealer due to bankruptcy ^^^246911!!!The IT organization does not have a strategy for the identification and implementation of technologies for automation of business process including the requirements definition, product analysis, procurement and implementation requirements resulting in poor business support, cost overruns and overall IT service issues.^^^246912!!!The organization does not have an overall continuity strategy including functional redundancy, system and infrastructure redundancy and recovery plans impacting the ability of the organization to recover from a business, service or technology disruption or disaster.^^^246920!!!The IT organization does not have properly documented and executed change management process for all application, infrastructure and internal and external IT services resulting in improperly documented or undocumented changes and unapproved untraceable or untested changes to production systems impacting IT resource availability and stability, confidentiality of data, data integrity and/or accountability of IT systems.^^^246921!!!Management lacks the ability to communicate clearly and effectively to the employees resulting in a lack of visibility to strategic goals and objectives, poor execution on strategic visions, tactical delivery failures and mis-represented management intentions.^^^246922!!!The IT organization does not have the capability to manage individual system and infrastructure configurations including implementation of new systems and maintenance of existing systems resulting in inconsistent control implementation, technical vulnerabilities (security, performance and availability), increased operational and maintenance costs.^^^246924!!!The organization has risks associated with delivery of products and/or services to end customers resulting in market share impact, loss of profits, loss of customer base, loss of market reputation or overall financial losses related to customer revenue.^^^246925!!!The organization does not have a data/information lifecycle process to define data management requirements including retention requirements, backup and recovery procedures, media library management functions and disposal of media increasing risks to the confidentiality, availability and integrity of corporate information and potential data related regulatory issues.^^^246928!!!Inadequate electronic informaition security management practices lead to breach of informaiton maintained by contractual third parties.^^^246935!!!The organization does not have the capability to manage areas of non-compliance to corporate policy through a risk based exception process resulting in unmitigated risks.^^^246936!!!The organization lacks an executive management sponsor for risk management leading to poor visibility at the business and corporate management level.^^^246939!!!The organization does not have proper oversight and management of external service providers including service level agreements, contractual definition of the service levels, procurement of services, financial oversight and relationship management resulting in poor or unmonitored performance of external parties, failure of internal services, inherited risks or civil and/or legal liability.^^^246941!!!Facilities used by the company have potential risks caused by the location, management processes or physical attributes of the facility itself resulting in environmental issues, loss of availability of the facility, interruption of business processes, loss of production capabilities or financial loss.^^^246943!!!The organization lacks the financial oversight function to manage budgetary and fiduciary responsibilities resulting in cost overruns, insufficient resources, exorbitant cost to the business and overall financial instability.^^^246944!!!Not all financial transactions will be recorded in the monthly financial statements.^^^246945!!!The organization does not have the capability to manage control failures or compliance issues through the lifecycle of identification, analysis, resolution and remediation resulting in unidentified control failures or unmitigated risks.^^^246972!!!The organization does not have a designated body for management of risks within functional areas such as information security, business continuity, Information Technology or lines of business leading to poor delivery of tactical risk management capabilities.^^^246973!!!  Losses due to the default on general obligation bonds ^^^246974!!!Management does not have the capability to establish business practices, monitor overall compliance to internal policy and communicate to board level management resulting in organizational risks such as regulatory non-compliance, missed strategic objectives and tactical operational issues.^^^246975!!!The organization lacks a defined process for the recruitment, retention and maintenance of appropriate skills and resources resulting in loss of intellectual knowledge, poor information sharing, overworked or underutilized resources and poor service and operational delivery.^^^246976!!!The organization does not have the capability to gather, track, respond and resolve incidents such as regulatory issues, end user issues, complaints, internal support requests and other business incidents resulting in customer dissatisfaction, recurring incidents and productivity loss.^^^246977!!!Failure to properly define and classify information, leading to accidental loss of information that would otherwise be properly classified.^^^246978!!!Employees do not follow approved information handling procudures as a result of inadequate education and training.^^^246979!!!Management does not have the capability to monitor the internal control environment including monitoring of operational security, application and data protection controls, controls for outsourced and third-party services or other internal controls leading to a poor understanding of internal control gaps and issues.^^^246983!!!The organization does not have a process for the definition of, documentation and agreement to internal business requirements for services and solutions resulting in inadequate business value, uninformed, dissatisfied or frustrated customers, gaps between business needs and IT services and overall poor relations with the business.^^^246985!!!The organization lacks an overall strategic direction for services, management, delivery or organizational structure leading to shifting goals, objectives, tactical delivery and strength of service capabilities.^^^246986!!!The IT organization does not have a defined process for installation, implementation, testing or delivery of IT resources (applications, infrastructure and services) resulting in potential issues with the stability and delivery of IT services.^^^246987!!!The concept of Least Privilege is not used leading to over-authorization of users' roles or access to data, transactions or business systems.^^^246988!!!The organization lacks the capability to manage legal affairs such as discovery acts, litigation support or contractual negotiations resulting in poor legal standing in contracts, potential litigation or criminal and civil actions.^^^247017!!!The organization does not have a management oversight function that is committed to risk management, fiduciary responsibilities or compliance management resulting in poor leadership representation to meet business needs.^^^247018!!!  Risk of non-compliance with anti-money laudering and bank secrecy act regulations ^^^247019!!!  Risk of non-compliance with the Foreign Corrupt Practices Act ^^^247020!!!The organization does not have the capability to capture, track, measure and evaluate operational costs leading impacting budget and financial metrics, costs to the business and cost-to-business value ratio.^^^247021!!!The organization has not created the proper operations infrastructure to manage business processes or assets on a daily operational basis including the proper documentation, monitoring and support tasks increasing the risk of an interruption of processes impacting business delivery services.^^^247022!!!The organization does not have a strategy including application performance and infrastructure utilization planning to properly make use of, and plan accordingly, IT resources resulting in IT resource (application and infrastructure) instability, cost overruns, increased operational maintenance and support costs, under-utilized resources and financial impacts.^^^247023!!!Management does not have the capability to monitor ongoing services and operations against define metrics to identify areas of improvement increasing costs and operational overhead.^^^247024!!!Loss of monies or assets due to the physical theft of vendors and their employees.^^^247025!!!Loss of monies or assets due to the physical theft of vendors and their employees.^^^247026!!!The organization does not have the capability to properly define corporate policy and standards requirements leading to poor overall governance and accountability for control and services management.^^^247027!!!The organization has risks associated with collection, processing or dissemination of personally identifiable information or information under the jurisdiction/purview of privacy regulations resulting in potential reputational, regulatory or financial losses.^^^247028!!!The organization does not have the capability to identify and investigate issues such as recurring incidents increasing operational costs, resource allocation conflicts and other systemic problems resulting in continued negative business, financial and service impacts.^^^247029!!!The organization does not have properly documented and executed procurement processes for resources including capital investments, hardware, software and services resulting in cost overruns, delay in service delivery, inadequate contractual support or coverage and potential legal or civil liability.^^^247032!!!The organization does not have an overall project management process including the definition, chartering, delivery and execution of projects resulting in improper allocation of resources, cost and budget overruns, delayed or late service delivery and customer dissatisfaction.^^^247033!!!Damage of buildings and other company assets from Earthquakes and sink holes.^^^247034!!!Damage of buildings and other company assets from Earthquakes and sink holes.^^^247035!!!Damage to properties from fire^^^247036!!!Damage to properties from fire^^^247037!!!Property damage due to water damage outside federally designated flood plain^^^247038!!!Property damage due to water damage outside federally designated flood plain^^^247039!!!Damage to property as a result of wind-related events.^^^247040!!!Damage to property as a result of wind-related events.^^^247041!!!The organization has not created the proper physical environment to protect information assets, data processing centers, inventories, production facilities and infrastructure devices increasing the risk of physical loss or harm to information  ^^^247042!!!The organization does not include quality assurance and management capabilities as a fundamental component of business practices or service delivery resulting in poor quality, increased costs due to reperformance of work and customer dissatisfaction.^^^247043!!!The organization is not aware of or does not commit to meeting regulatory requirements including legislative and/or industry obligations resulting in non-compliance that can result in fines, censures, civil and legal liabilities.^^^247046!!!The organization does not have the capability to identify, assess, manage and control risks at the strategic and tactical levels resulting in inadequate understanding of the business impact, control requirements and requirements for delivery of sustainable services.^^^247048!!!The IT organization does not have the capability to provide protection of IT systems including access to resources and protection from internal and external threats impacting the confidentiality, integrity and availability of business data and/or information systems.^^^247049!!!The organization does not have the capability to segregate duties within operational personnel such that conflicting responsibilities are performed by different individuals allowing for accountability and oversight.^^^247050!!!Applications or business systems do not have the capability to segregate user privileges, transactions or functions such that conflicting end user actions cannot be denied or restricted allowing for accountability, confidentiality, integrity or availability of data.^^^247051!!!The IT organization does not properly document and communicate the use and operation of IT resources (applications, infrastructure, etc.) resulting in poor end user experiences, increased support costs, increased help desk incidents and uninformed maintenance and operations support personnel.^^^247052!!!The IT organization does not have the capability to operationally support technology infrastructure (networks, operational platforms, etc.) over the life of the technology from definition to development to implementation to retirement resulting in improper software acquisition, costly legacy systems, burdensome maintenance and support requirements and poor IT service delivery.^^^247053!!!The IT organization does not have a defined technology strategy and architecture resulting in inherent technological risks such as poor technology acquisitions, over or under utilized technology resources, disparate or disconnected systems and processes.^^^247056!!!Employee theft of company assets, intellectual property, or customer information for personal gain or other purposes. ^^^247057!!!Employee theft of company assets, intellectual property, or customer information for personal gain or other purposes. ^^^247058!!!Electronic breach of firewalls results in loss of company intellecual property.^^^247059!!!Electronic breach of firewalls results in loss of company intellecual property.^^^247060!!!Litigation claims associatied with customers and third parties being injured on company owned or rented properties due do maintenance, upkeep, or other liabilities.  Includes slip and falls.^^^247061!!!Litigation claims associatied with customers and third parties being injured on company owned or rented properties due do maintenance, upkeep, or other liabilities.  Includes slip and falls.^^^247062!!!The IT organization does not enforce policy around application development to ensure that unauthorized access to information can not be accomplished through internet-facing application software SQL injection attacks^^^281029!!!  The risk of loss from stolen merchandise, internally or externally. ^^^281031!!!   Loss of revenue as a result of business interruptions from natural or manmade disaster, vandalism or equipment malfunction. ^^^281080!!!  Revenue from the sale of widgets is not recorded properly as a result of collecting the wrong amount or entering it into the sales register incorrectly ^^^281193!!!   Loss of employee productivity and injury / possible litigation or workers compensation claims from employees, or from customers or others on-premises. ^^^353149!!!  During our internal reviews, and based on some findings from the internal audit department, we recognized that there were active accounts in a customer service application containing over 10,000 records with inappropriate access privileges. ^^^353167!!!As an enterprise primarily reliant on ecommerce sales, the board has asked us to assess our capabilities to defend against a denial of service (DOS) attack on our main website where the majority of our sales revenue is generated.^^^356199!!!Due to a total power outage, our primary data center goes down. Back Up power systems do not engage, resulting in complete failure of the system.^^^356306!!!Due to recent security events reported in the news, the board has asked for us to review access privileges of external 3rd parties into internal systems. This security review found a number of 3rd party credentials have been left open and unmonitored.^^^356307!!!Due to impending international regulations around data privacy, like GDPR, Legal has required us to analyze the risk of failing a Regulator Audit.^^^356308!!!Employee intranet polls indicated an elevated concern of physical safety due to perceived lack of physical access controls to corporate headquarters. Unauthorized access could result in physical harm, harrassment, and/or theft of corporate assets,^^^359374!!!  Operational fraud, loss of intellectual property, and loss of customers from damaged reputation resulting from an access control breach ^^^359377!!!Description 2^^^361403!!!  asdf ^^^361404!!!  encryption ^^^361405!!!  ttt "
    query_description="financial"
    descs=corporaEncoded.split("^^^")
    for i in range(len(descs)):
        content=descs[i].split("!!!")
        corpora[int(content[0])]=content[1]
            
    proximityFinder= Proximity_finder_for_documents(corpora)

    results=proximityFinder.find_proximity(query_description)
    if(results==" "):
        print("  ")
    else:
        indices='^^^'.join(str(x) for x in results)
        print(indices)
   

if __name__ == "__main__":   

       main()
   
