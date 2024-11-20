import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_data = [
    {
        "question": "1. I received my provisional from an autonomous college, affiliated to JNTUH. Am I eligible to apply for transcripts?",
        "answer": "No, colleges which are affiliated and non-autonomous, are only eligible for applying transcripts in online student service."
    },
    {
        "question": "2. When I tried to apply for a online service in online student service portal, \nI am getting a response -invalid hall ticket number or data not available. Why?",
        "answer": "The Candidates who passed their degree during or after the academic year mentioned below can only avail this service.\nB.Tech?2000 to till date,\nB.Pharmacy- 2009 batch to till date,\nM.Tech ? 2009 batch to till date,\nM.Pharmacy ?2009 batch to till date,\nMBA ? 2005 batch to till date,\nMCA - 2005 batch to till date."
    },
    {
        "question": "3. Am I eligible to apply in online student service?",
        "answer": "Yes, all registered students of JNTUH are eligible to use the Online Student Services. The services include accessing academic details, applying for transcripts, revaluation, obtaining certificates, and checking results. To access these services, students need to log in using their hall ticket or student credentials on the official JNTUH Online Student Services Portal."
    },
    {
        "question": "1. How to apply for online student service?",
        "answer": "Open the JNTUH website (jntuh.ac.in) and click on the online student service link at the home page. It will redirect to online student service portal. (OR) by using the url:http://studentservices.jntuh.ac.in/oss."
    },
    {
        "question": "2. How to apply online transcript for Original Degree certificate?",
        "answer": "At present, there is no transcript service for Original Degree certificate."
    },
    {
        "question": "3. How to apply online transcript for Original Degree certificate?",
        "answer": "At present, there is no online transcript service for Original Degree certificate."
    },
    {
        "question": "1. For any clarification or information regarding transcripts, whom should I contact?",
        "answer": "You should contact JNTUH Exam Branch at Email:support.oss@jntuh.ac.in or at helpline no. 9491283135 during 10:30 AM to 5:00 PM on all working days."
    },
    {
        "question": "2. What are the services offered under Student Services?",
        "answer": "1. Manual Services at counters of Service Section, Examination Branch.\n2. Tatkal Services at counters of Service Section, Examination Branch.\n3. Online Student Services."
    },
    {
        "question": "3. What are the Manual Services offered at counters of Service Section, Examination Branch?",
        "answer": "Issuing of Duplicate Marks Memos, Duplicate CMM, Duplicate Degree Certificate, Name &  Father Name correction in PC & OD Certificates, \nName corrections in marks memo & CMM,  Issuing of Transcripts, Issuing of PC & CMM with Undertaking and Grace Marks."
    },
    {
        "question": "4. What are the Tatkal Services offered at counters of Service Section, Examination Branch?",
        "answer": "Tatkal PC & CMM, Tatkal PC & CMM with undertaking for B.Tech. Only, Tatkal CMM,\nDuplicate CMM, Tatkal Name and Gender Correction in PC.CMM/Marks Memo"
    },
    {
        "question": "5. What are the Online Services offered?",
        "answer": "Issuing of Original Degree Certificate, Migration Certificate, Medium of Instruction\nCertificate and issuing Transcripts along with WES Application"
    },
    {
        "question": "6. Where can we have the application forms for availing the Student Services?",
        "answer": "Under:  Application Forms  of  https://studentservices.jntuh.ac.in/oss/login page."
    },
    {
        "question": "7. Who are eligible to opt the various services offered?",
        "answer": "1. Manual Services - The Candidates who passed their degree before the academic year\nMentioned below can only avail this service. Affiliated colleges degree of  B.Tech. - Before\n2000  admitted batch, B.Pharmacy - before 2009 batch, M.Tech. ? before 2009 batch,\nM.Pharmacy ? before 2009 batch, MBA ? before 2005 batch, MCA ? before 2005 batch.\n2. Online Services : The Candidates who passed their degree during or after the academic\n year mentioned below can only avail this service. B.Tech. - 2000 to till date, B.Pharmacy-\n 2009 batch to till date, M.Tech. ? 2009 batch to till date, M.Pharmacy ?2009 batch to till date,\n MBA ? 2005 batch to till date, MCA - 2005 batch to till date.\n3. Tatkal Services : As specified in the application form."
    },
    {
        "question": "8. Is this service offered for all colleges under JNTUH?",
        "answer": "1. The students of Autonomous colleges under JNTUH can avail manual services.\n2.  Only students of colleges affiliated and non-autonomous to JNTUH are eligible to avail online\nServices"
    },
    {
        "question": "9. What are the payment modes for applying?",
        "answer": "The payments can be made only in Online-payment mode for online transactions The \npayments for other services can be made in one of the following modes : Smart-card swiping/ \nT-Wallet/SBI Challan / DD. If Demand Draft is taken, it should be drawn in favor of ?THE\n REGISTRAR JNTUH? , payable at Hyderabad. The student Hall-Ticket number should be\n written on the Backside of DD. Student should write his/her Hall Ticket Number on the back-\nside of the Demand Draft, if the mode of payment is DD. If the student desires to choose \nchallan option, the Challan should be taken only at the campus SBI (JNTU Hyderabad)\n Branch. The T-Wallet/ Swiping facility using the smart card (debit/credit card) is available at\n the student service counter.Original Degree (OD) certificate will be sent only by speed-post to\n the postal address, which was mentioned while applying for the OD. (OD Application service \nfor 'Passed out candidates after 2002' will be available\nat https://epayments.jntuh.ac.in/onlineconvocationservices)."
    },
    {
        "question": "10. After applying manually/ Tatkal (by hand) within how many days can I collect my certificates from\n service section counter of  exam branch JNTUH?",
        "answer": "The candidates are given acknowledgement slip mentioning the date to collect their \ncertificates from the counters of Service Section at exam branch, JNTUH. Certificates under\n TATKAL scheme will be issued within in two working days. The transcripts, migration\n certificate and Medium of Instruction certificate can be either delivered by post or collected \nby hand at student service counter"
    },
    {
        "question": "11. I have applied for transcripts and received SMS/email to collect them from University counter. \nWhat identification proof should I bring for receiving my certificates?",
        "answer": "You should confirm your registered mobile number (which is used during registration) with\n dispatch clerk at the exam branch Student Service Counters .No, other proof is required."
    },
    {
        "question": "12. What are different stages until the consignment reaches my residential address?",
        "answer": "PAYMENT APPROVED - After payment approved,\nPRINTING - Printing of applied certificates \nINPROCESS - checking of applied certificates with address slip \nDISPATCHEDBYPOST ? Dispatched the applied certificates by post\n DISPATCHEDBYHAND ? Dispatched the applied certificates to service counter\n DELIVERED - the applied certificates received by the candidate."
    },
    {
        "question": "13. What are the modes of dispatching the certificates?",
        "answer": "Local consignments are sent by Speed post or DTDC courier, International consignments are\n sent by Blue Dart Services"
    },
    {
        "question": "14. For any clarification or information regarding transcripts, whom should I contact?",
        "answer": "You should contact JNTUH Exam Branch at Email:support.oss@jntuh.ac.in or at helpline\n no. 9491283135 during 10:30 AM to 5:00 PM on all working days."
    },
    {
        "question": "1. What to do if we get a message saying transaction failed / session expired?",
        "answer": "If your amount were deducted from your account in any transaction and got failed / session \nexpired message, You are advised not to pay once again. The application status will be\n updated in 'Track Application' menu within 2 working days of your transaction."
    },
    {
        "question": "2. What should we do if amount is deducted many times (multiple payments)?",
        "answer": "Users should not pay fee multiple times for one request. Because every transaction will take\n as request and service will be provided for every transaction.\n\nIf your amount were deducted from your account in any transaction and were failed / session\n expired message, You are advised not to pay once again. The application status will be \nupdated in 'Track Application' menu within 2 working days of your transaction."
    },
    {
        "question": "3. What is the procedure for refund?",
        "answer": "No refund will be entertained. Every transaction will be serviced by request transcripts / certificates.\nBecause, the process of your requested transcripts/ certificates printing is started with in few time, \nprinted transcripts / certificates can?t be used for another application, Stationary will be waste.\nYou are advised not to pay once again if you already got deducted money. \nThe application status will be updated in 'Track Application' menu within 2 working days of your transaction"
    },
    {
        "question": "1. I have forgotten my online student service portal password. How can I get my password?",
        "answer": "Click on forgot password link available at the home page of online student service portal. It will ask to enter your hall ticket number, registered mobile number & email-id for verification. Once the verification is successful, you will receive your password to your registered email-id."
    },
    {
        "question": "2. I have forgotten my online student service portal password. How can I get my password?",
        "answer": "Click on forgot password link available at the home page of online student service portal.\n It will to enter your hall ticket number, registered mobile number & email-id for verification. \nOnce the verification is successful, you will receive your password to your registered email-id."
    },
    {
        "question": "1. After applying (preferred mode: by hand) within how many days can I collect my consignment from JNTUH exam branch service counter?",
        "answer": "The candidates are informed to collect their transcripts from the service counters at exam branch JNTUH, within seven days after receiving the sms/email. If not collected within\nseven days, the candidate has to forego the claim of the transcripts. Under any circumstances, the transcripts will not be issued after seven days of receiving the sms/email."
    },
    {
        "question": "2. How long it takes to get my Certificates after applying in online student service?",
        "answer": "The printing and dispatching process will be completed within 2 working days after the payment is approved."
    },
    {
        "question": "3. Can I apply for online student service in tatkal service?",
        "answer": "There is no tatkal service to online student service."
    },
    {
        "question": "4. I have lost my Acknowledgement receipt. Where can I get it ?",
        "answer": "It is also sent to your registered email id, Hence you can login to your mail and can take the printout whenever you require."
    },
    {
        "question": "5. I got an Acknowledgement. What should I do after that?",
        "answer": "You keep that printout for further reference. In case if you have not received the applied certificates, then you can contact, Exam Branch, JNTUH with the printout of acknowledgement, at the student service counter no 1."
    },
    {
        "question": "6. What are the other certificates, in addition to transcript, that I can obtain using this portal service?",
        "answer": "You can also apply for migration and medium certificate an additional to transcripts."
    },
    {
        "question": "7. As I am trying to track using the consignment number, it shows the consignment number given is not valid.What should I do?",
        "answer": "Consignment number received by you is correct. It will be trackable once postal/courier service picks and updates their database. You are advised to try again at the respective the postal / courier portals on the following day."
    },
    {
        "question": "8. I have applied for transcripts and received SMS/email to collect them from University counter. What identification proof should I bring for receiving my certificates?",
        "answer": "You should confirm your registered mobile number (which is used during registration) with dispatch clerk at the exam branch student service .No other proof is required ."
    },
    {
        "question": "9. What are different stages until the consignment reaches my residential address?",
        "answer": "PAYMENT APPROVED-After payment approved\nPRINTING- Printing of applied certificates\nINPROCESS-checking of applied certificates with address slip\nDISPATCHEDBYPOST ? Dispatched the applied certificates by post\nDISPATCHEDBYHAND ? Dispatched the applied certificates to service counter\nDELIVERED - the applied certificates received by the candidate."
    },
    {
        "question": "10. How long does it take to get my Certificates after applying in online student service?",
        "answer": "The printing and dispatching process will be completed within 2 working days after the\n payment is approved, time to time you will receive message / mail once it is ready to dispatch\n from JNTUH"
    },
    {
        "question": "11. Can I apply for online student service in tatkal service?",
        "answer": "There is no tatkal service to online student service."
    },
    {
        "question": "12. I have lost my Acknowledgement receipt. Where can I get it?",
        "answer": "It is also sent to your registered email id, hence you can login to your mail and can take the\n printout whenever you require. The receipt should be preserved for future reference"
    },
    {
        "question": "13. As I am trying to track using the consignment number, it shows the consignment number given is not valid. What should I do?",
        "answer": "Consignment number received by you is correct. It will be trackable once postal/courier\n service picks and updates their database. You are advised to try again at the respective the\n postal / courier portals on the following day."
    },
        {
            "question": "How do I apply for my Original Degree (OD) through JNTUH?",
            "answer": "To apply for an OD, follow these steps:\n1. Visit the JNTUH Convocation Portal at https://epayments.jntuh.ac.in/onlineconvocationservices.\n2. Register with your hall ticket number, date of birth, and valid email ID.\n3. Fill in the required details, upload necessary documents, and make the payment.\n4. After submission, you will receive an acknowledgment receipt. Keep it for future reference."
        },
        {
            "question": "I have forgotten my online student service portal password. How can I recover it?",
            "answer": "To recover your password:\n1. Click on the 'Forgot Password' link available on the login page of the Online Student Service Portal.\n2. Enter your hall ticket number, registered mobile number, and email ID for verification.\n3. Once verified, you will receive your password on your registered email ID."
        },
        {
            "question": "What should I do if I see a message saying 'Transaction Failed' or 'Session Expired' while applying?",
            "answer": "If your transaction fails or session expires, but the amount is deducted, do not make another payment. The application status will be updated in the 'Track Application' menu within two working days. If the issue persists, contact the support team."
        },
        {
            "question": "Can I apply for transcripts online if I studied in an autonomous college affiliated with JNTUH?",
            "answer": "No, only students from affiliated and non-autonomous colleges under JNTUH are eligible to apply for transcripts online. Students from autonomous colleges can avail manual services at the JNTUH Exam Branch."
        },
        {
            "question": "What identification proof should I bring when collecting certificates from the JNTUH counter?",
            "answer": "You should confirm your registered mobile number used during the registration process with the dispatch clerk at the Exam Branch Student Service Counters. No other proof is required."
        },
        {
            "question": "What are the payment methods for applying to JNTUH student services?",
            "answer": "Payments can be made using the following methods:\n1. Online payment (for online services).\n2. Smart-card swiping, T-Wallet, SBI Challan, or Demand Draft (for manual services).\nIf using a Demand Draft, it should be in favor of 'THE REGISTRAR JNTUH,' payable at Hyderabad, with the hall ticket number written on the backside."
        },
        {
            "question": "What are the services offered under the JNTUH Student Services?",
            "answer": "JNTUH Student Services include:\n1. Manual Services (e.g., issuing duplicate marks memos, name corrections).\n2. Tatkal Services (e.g., expedited PC & CMM services).\n3. Online Services (e.g., issuing OD certificates, transcripts, migration certificates)."
        },
        {
            "question": "How long does it take to receive my certificates after applying through the online student service?",
            "answer": "The printing and dispatching process is usually completed within two working days after the payment is approved. You will receive updates via SMS or email during each stage of processing."
        },
        {
            "question": "What should I do if my consignment tracking number shows as invalid?",
            "answer": "The consignment number provided is correct. It becomes trackable once the postal or courier service updates their database. Please try tracking again after a day at the respective postal or courier portal."
        },
        {
            "question": "Can I apply for Tatkal service using the Online Student Service Portal?",
            "answer": "No, Tatkal services are not available through the Online Student Service Portal. Tatkal services can only be availed manually at the JNTUH Exam Branch."
        },
        {
            "question": "What are the steps involved in the certificate dispatch process?",
            "answer": "The steps in the dispatch process are as follows:\n1. PAYMENT APPROVED - Payment confirmed.\n2. PRINTING - Certificates are printed.\n3. IN PROCESS - Certificates are checked with the address slip.\n4. DISPATCHED BY POST - Certificates are sent via post.\n5. DISPATCHED BY HAND - Certificates are available at the service counter.\n6. DELIVERED - Certificates are received by the candidate."
        },
        {
            "question": "What are the modes of dispatch for certificates?",
            "answer": "Local consignments are sent via Speed Post or DTDC Courier. International consignments are sent using Blue Dart Services."
        },
        {
            "question": "How do I contact JNTUH for clarifications regarding transcripts?",
            "answer": "For clarifications, you can contact the JNTUH Exam Branch at the following:\nEmail: support.oss@jntuh.ac.in\nHelpline: 9491283135 (Available from 10:30 AM to 5:00 PM on all working days)."
        },
        {
            "question": "Where can I download the application forms for manual student services?",
            "answer": "Application forms for manual services are available on the Online Student Service Portal at https://studentservices.jntuh.ac.in/oss/login under the 'Application Forms' section."
        },
        {
            "question": "What are the eligibility criteria for different student services?",
            "answer": "Eligibility varies as follows:\n1. Manual Services: For candidates who passed their degree before specific academic years (e.g., B.Tech before 2000 batch).\n2. Online Services: For candidates who passed their degree during or after specific academic years (e.g., B.Tech from 2000 onwards).\n3. Tatkal Services: Eligibility is as specified in the application form."
        },
    {
        "question": "How do I track the status of my application after submission?",
        "answer": "You can track your application status through the 'Track Application' menu in the Online Student Service Portal. Enter your application ID or hall ticket number to view the status updates."
    },
    {
        "question": "What documents are required to apply for transcripts at JNTUH?",
        "answer": "To apply for transcripts, you need the following documents:\n1. Scanned copy of consolidated marks memo (CMM) or individual semester memos.\n2. Payment receipt.\n3. Application acknowledgment receipt from the online portal."
    },
    {
        "question": "What should I do if my name is misspelled on my certificates?",
        "answer": "For name correction:\n1. Apply manually at the JNTUH Exam Branch with a letter of request.\n2. Submit proof of the correct name (such as Aadhaar or SSC certificate).\n3. Pay the required fee using the prescribed payment methods."
    },
    {
        "question": "What is the process for applying for a Migration Certificate?",
        "answer": "To apply for a Migration Certificate:\n1. Login to the Online Student Service Portal.\n2. Select 'Migration Certificate' from the list of services.\n3. Upload the required documents, including a copy of your provisional certificate (PC).\n4. Make the necessary payment and submit the application."
    },
    {
        "question": "How can I cancel an application submitted through the online portal?",
        "answer": "Applications submitted online cannot be canceled. If there are errors, you must reapply and make a new payment. Contact the support team for further guidance if needed."
    },
    {
        "question": "What are the fees for applying for duplicate marks memos at JNTUH?",
        "answer": "The fees for duplicate marks memos vary. Typically, it is Rs. 100 per memo. Check the latest fee details on the official JNTUH website or the application form for manual services."
    },
    {
        "question": "How can I apply for my Provisional Certificate (PC) after my results are released?",
        "answer": "To apply for a Provisional Certificate (PC):\n1. Login to the Online Student Service Portal.\n2. Select 'Provisional Certificate' service.\n3. Upload the scanned copies of your final semester marks memo and other required documents.\n4. Pay the application fee and submit the form."
    },
    {
        "question": "Can I collect my certificates in person instead of waiting for postal delivery?",
        "answer": "Yes, you can collect your certificates in person if the status shows 'DISPATCHED BY HAND.' Visit the JNTUH Exam Branch Student Service Counters during working hours and confirm your registered mobile number."
    },
    {
        "question": "What should I do if my postal address is incorrect in the application?",
        "answer": "You can update your postal address in the 'Track Application' section before the application reaches the 'DISPATCHED' stage. If already dispatched, you will need to contact the postal service to redirect the delivery."
    },
    {
        "question": "What is the difference between consolidated marks memo (CMM) and semester marks memo?",
        "answer": "A consolidated marks memo (CMM) contains the overall grades/marks for all semesters, while a semester marks memo includes grades/marks for a specific semester only."
    },
    {
        "question": "Are there any priority services for obtaining certificates from JNTUH?",
        "answer": "Yes, JNTUH offers Tatkal services for urgent requirements. These services are available only at the Exam Branch and typically process your application within 24-48 hours."
    },
    {
        "question": "What happens if my payment fails during the application process?",
        "answer": "If your payment fails, the amount will be refunded automatically within 7-10 business days. You can retry the application after verifying the payment status in the portal."
    },
    {
        "question": "Is there any validity period for the Provisional Certificate (PC)?",
        "answer": "Yes, the Provisional Certificate (PC) is valid until the Original Degree (OD) is issued. Once you receive the OD, the PC is no longer considered valid."
    },
    {
        "question": "What should I do if my OD certificate is damaged or lost?",
        "answer": "To replace a damaged or lost OD:\n1. Apply for a duplicate OD through the manual process.\n2. Submit a notarized affidavit, a copy of the FIR (if lost), and proof of payment.\n3. Visit the JNTUH Exam Branch for further processing."
    },
    {
        "question": "Can I apply for more than one service at a time through the online portal?",
        "answer": "No, the portal allows only one service request per application. For multiple services, you need to submit separate applications and payments for each service."
    },
    {
        "question": "How do I apply for semester-wise grade sheets for WES or other evaluations?",
        "answer": "You can apply for semester-wise grade sheets through the 'Transcripts' service on the Online Student Service Portal. Ensure that you specify the requirement for semester-wise sheets in the remarks section."
    },
    {
        "question": "What is the procedure for obtaining a Genuineness Certificate from JNTUH?",
        "answer": "To get a Genuineness Certificate:\n1. Submit a manual application at the Exam Branch.\n2. Attach the photocopies of the relevant certificates and pay the prescribed fee.\n3. The certificate will be processed and dispatched after verification."
    },
    {
        "question": "What is the contact number for JNTUH Exam Branch support?",
        "answer": "For any queries, you can reach JNTUH Exam Branch support at 9491283135. The helpline is available from 10:30 AM to 5:00 PM on all working days."
    },
    {
        "question": "Can I apply for services through the portal if I have completed my degree from an autonomous college under JNTUH?",
        "answer": "No, the Online Student Service Portal is for students of affiliated and non-autonomous colleges only. Autonomous college students must apply manually at the Exam Branch."
    },
    {
        "question": "What are the working hours of the JNTUH Exam Branch for manual services?",
        "answer": "The working hours for manual services at the JNTUH Exam Branch are from 10:30 AM to 5:00 PM on all working days, excluding public holidays."
    },
        {
            "question": "What is the process for applying for a duplicate hall ticket?",
            "answer": "To apply for a duplicate hall ticket, you need to visit the JNTUH Exam Branch and submit a request letter along with proof of identity, a copy of the FIR if lost, and the required payment receipt."
        },
        {
            "question": "How do I reset my password for the Online Student Service Portal?",
            "answer": "To reset your password, click on the 'Forgot Password' link on the login page. Enter your registered email ID and follow the instructions sent to your email to create a new password."
        },
        {
            "question": "What should I do if I find an error in my grade sheet?",
            "answer": "If you find an error in your grade sheet, submit a request for correction to the JNTUH Exam Branch. Attach a copy of the incorrect grade sheet, relevant supporting documents, and a request letter."
        },
        {
            "question": "Can I get my original degree (OD) during the convocation ceremony?",
            "answer": "Yes, if you apply for the OD before the convocation deadline and are eligible, you can receive it during the ceremony. Otherwise, it will be dispatched to your address."
        },
        {
            "question": "What is the fee for the Tatkal service for a Provisional Certificate (PC)?",
            "answer": "The fee for the Tatkal service for a PC is typically Rs. 1000. This ensures faster processing within 1-2 working days."
        },
        {
            "question": "How long does it take to get my Original Degree (OD) after applying?",
            "answer": "It generally takes 4-6 weeks for the Original Degree (OD) to be dispatched after successful application. However, the timeline may vary based on processing loads."
        },
        {
            "question": "Can I make changes to my application after submitting it online?",
            "answer": "No, changes cannot be made to an application after submission. You must cancel the application (if allowed) and reapply with the correct details."
        },
        {
            "question": "What is the eligibility criteria for applying for a Provisional Certificate (PC)?",
            "answer": "You must have cleared all your subjects in the final semester and received your Consolidated Marks Memo (CMM) to be eligible for a PC."
        },
        {
            "question": "How can I verify the genuineness of my JNTUH degree certificates?",
            "answer": "You can apply for a Genuineness Verification Certificate through the JNTUH Exam Branch. This is typically requested by employers or institutions for verification."
        },
        {
            "question": "Can I download a digital copy of my marks memo online?",
            "answer": "No, JNTUH does not currently provide digital copies of marks memos. You must apply for physical copies through the Online Student Service Portal or manually."
        },
        {
            "question": "What documents are required to apply for a consolidated marks memo (CMM)?",
            "answer": "To apply for a CMM, you need:\n1. Scanned copies of individual semester grade sheets.\n2. Payment receipt.\n3. Application acknowledgment from the portal."
        },
        {
            "question": "How do I know if my application has been dispatched?",
            "answer": "You can check the dispatch status of your application in the 'Track Application' section of the Online Student Service Portal."
        },
        {
            "question": "Is there a penalty for delayed submission of applications for certificates?",
            "answer": "No specific penalty is charged for delayed applications. However, urgent requirements may necessitate using the Tatkal service, which has higher fees."
        },
        {
            "question": "What is the official website for JNTUH student services?",
            "answer": "The official website for JNTUH student services is [https://studentservices.jntuh.ac.in](https://studentservices.jntuh.ac.in)."
        },
        {
            "question": "Can I authorize someone else to collect my certificates on my behalf?",
            "answer": "Yes, you can authorize someone to collect your certificates by providing a signed authorization letter, a copy of your ID, and the authorized person's ID proof."
        },
        {
            "question": "What is the procedure for applying for a Rank Certificate?",
            "answer": "To apply for a Rank Certificate:\n1. Visit the JNTUH Exam Branch or check the online portal.\n2. Submit proof of your rank (if applicable).\n3. Pay the required fee and submit the application."
        },
        {
            "question": "What are the reasons for rejection of an application in the online portal?",
            "answer": "Applications may be rejected for reasons such as:\n1. Incorrect or missing documents.\n2. Payment discrepancies.\n3. Ineligibility for the requested service.\n4. Errors in the application details."
        },
        {
            "question": "How can I get duplicate certificates for documents damaged in a natural disaster?",
            "answer": "You must submit an affidavit, proof of the disaster (such as a government report), and a request letter along with the application fee for duplicate certificates at the JNTUH Exam Branch."
        },
        {
            "question": "What is the fee for revaluation of exam papers at JNTUH?",
            "answer": "The revaluation fee is typically Rs. 1000 per subject. Check the official notification for updated details and payment deadlines."
        },
        {
            "question": "How do I correct errors in my date of birth on my JNTUH certificates?",
            "answer": "To correct errors in your date of birth, submit a request letter, proof of correct date of birth (such as a birth certificate or SSC memo), and pay the required fee at the Exam Branch."
        },
        {
            "question": "Can I apply for services if I have backlogs in my exams?",
            "answer": "You cannot apply for certain services like a Provisional Certificate or CMM until you clear all your backlogs. However, you can apply for duplicate or semester-wise grade sheets."
        },
        {
            "question": "What is the procedure for obtaining a No Objection Certificate (NOC)?",
            "answer": "To get an NOC:\n1. Visit the JNTUH Exam Branch or affiliated college.\n2. Submit a request letter and necessary documents such as ID proof and a transfer request.\n3. Pay the applicable fee."
        },
        {
            "question": "What is the mode of payment for services on the Online Student Service Portal?",
            "answer": "Payments can be made using credit/debit cards, net banking, or UPI through the portal's integrated payment gateway."
        },
        {
            "question": "Can I apply for an OD if my college has not issued my CMM?",
            "answer": "No, the Consolidated Marks Memo (CMM) is a prerequisite for applying for the Original Degree (OD). Contact your college for assistance in obtaining your CMM."
        },
        {
            "question": "How do I escalate an issue if my application is delayed or rejected unfairly?",
            "answer": "You can escalate the issue by sending an email to support services or visiting the JNTUH Exam Branch in person with all relevant documents and proof of your application."
        },
        {
            "question": "What is the timeline for receiving a duplicate marks memo?",
            "answer": "Duplicate marks memos are usually processed within 2-4 weeks of application. Delays may occur during peak periods or due to incomplete documentation."
        },
        {
            "question": "Is there a way to verify the status of revaluation results online?",
            "answer": "Yes, you can check the revaluation results on the official JNTUH results website. Enter your hall ticket number and the relevant exam details."
        },
        {
            "question": "What should I do if I lose my application ID for tracking purposes?",
            "answer": "If you lose your application ID, you can retrieve it by logging into the portal and checking your application history under your profile."
        },
        {
            "question": "Does JNTUH provide internship-related certifications?",
            "answer": "No, JNTUH does not directly provide internship certifications. Students need to obtain such certificates from the organizations where they complete their internships."
        },
        {
            "question": "Can I update my contact details in the Online Student Service Portal?",
            "answer": "Yes, you can update your email address and phone number in the profile section of the portal. Ensure that these details are up-to-date to avoid communication issues."
        },
          {
                "question": "What is the process for obtaining a transcript for higher education abroad?",
                "answer": "To obtain a transcript, apply through the Online Student Service Portal or directly at the JNTUH Exam Branch. Submit copies of all marks memos, a request letter, and the required fee. Transcripts are sealed and signed for authenticity."
            },
            {
                "question": "How do I report discrepancies in exam question papers?",
                "answer": "Discrepancies in question papers should be reported immediately through your college exam coordinator. They will escalate the issue to JNTUH for resolution."
            },
            {
                "question": "What is the fee for applying for a Medium of Instruction (MOI) certificate?",
                "answer": "The fee for an MOI certificate is typically Rs. 200. Confirm the exact fee on the Online Student Service Portal or through your college."
            },
            {
                "question": "Can I apply for revaluation after the revaluation deadline has passed?",
                "answer": "No, revaluation applications are not accepted after the deadline. Ensure you apply within the stipulated timeline mentioned in the notification."
            },
            {
                "question": "How can I check my backlog status online?",
                "answer": "You can check your backlog status by logging into the JNTUH Results Portal. Enter your hall ticket number to view detailed subject-wise results."
            },
            {
                "question": "What documents are needed for a name correction on certificates?",
                "answer": "For name corrections, submit:\n1. A request letter.\n2. A copy of ID proof with the correct name.\n3. Affidavit or gazette notification.\n4. Original certificate needing correction."
            },
            {
                "question": "What are the academic rules for promotion to the next year?",
                "answer": "To be promoted, students must meet the minimum credit requirements for their respective academic year. Typically, students need to clear at least 50% of their subjects to advance."
            },
            {
                "question": "Is there a limit on the number of times I can apply for revaluation?",
                "answer": "There is no specific limit on the number of revaluation applications. However, you can apply for a subject's revaluation only for the most recent attempt."
            },
            {
                "question": "How do I apply for a duplicate hall ticket before exams?",
                "answer": "Visit your college administration office and submit a request for a duplicate hall ticket. Provide proof of identity, a passport-size photo, and a small fee (if applicable)."
            },
            {
                "question": "What is the grading system used at JNTUH?",
                "answer": "JNTUH uses a 10-point grading system based on marks scored in each subject. Grades range from O (Outstanding) for scores of 90-100 to F (Fail) for scores below 40."
            },
            {
                "question": "How can I download the syllabus for my course online?",
                "answer": "You can download the syllabus from the official JNTUH website under the Academics section. Choose your program and year to access the relevant syllabus."
            },
            {
                "question": "Can I apply for an original degree (OD) before completing my course?",
                "answer": "No, the Original Degree (OD) can only be applied for after completing your course and receiving your Consolidated Marks Memo (CMM)."
            },
            {
                "question": "What is the procedure for obtaining a migration certificate?",
                "answer": "To obtain a migration certificate, apply through the Online Student Service Portal. Submit a request letter, a copy of the provisional certificate or degree, and pay the required fee."
            },
            {
                "question": "How long are revaluation results typically delayed?",
                "answer": "Revaluation results are generally released within 3-4 weeks of the application deadline. Delays may occur during peak examination periods."
            },
            {
                "question": "Can I change my specialization after enrolling in a program?",
                "answer": "No, specialization changes are generally not permitted once a program has started. You must complete the program in your chosen specialization."
            },
            {
                "question": "What is the fee for issuing duplicate grade sheets?",
                "answer": "The fee for duplicate grade sheets is Rs. 200 per sheet. Additional charges may apply for Tatkal processing if requested."
            },
            {
                "question": "How do I apply for supplementary exams?",
                "answer": "Supplementary exam applications can be submitted through the JNTUH Exam Portal. Pay the required exam fee for each subject and download your hall ticket once issued."
            },
            {
                "question": "What is the validity period for a provisional certificate (PC)?",
                "answer": "The provisional certificate is valid until the Original Degree (OD) is issued. It serves as temporary proof of graduation."
            },
            {
                "question": "How do I update my email address in the JNTUH portal?",
                "answer": "To update your email address, log into the Online Student Service Portal, navigate to the profile settings, and edit your contact details. Verify the new email address to save changes."
            },
            {
                "question": "Can I attend the convocation ceremony without prior registration?",
                "answer": "No, you must register for the convocation ceremony in advance to receive an invitation and participate. Registration is typically done through the Online Student Service Portal."
            },
            {
                "question": "What is the procedure for withdrawing from a course at JNTUH?",
                "answer": "To withdraw from a course, submit a withdrawal application through your college. Provide valid reasons, supporting documents, and approval from the Head of the Department (HOD)."
            },
            {
                "question": "What happens if I fail to attend a scheduled exam?",
                "answer": "If you fail to attend an exam, it will be considered an absent attempt, and you will need to reapply for the supplementary exam to clear the subject."
            },
            {
                "question": "How do I apply for a refund if I make an incorrect payment on the portal?",
                "answer": "Submit a refund request with proof of payment and a valid reason to the JNTUH Exam Branch. Refunds are typically processed within 15-30 days if approved."
            },
            {
                "question": "Does JNTUH provide a consolidated attendance report?",
                "answer": "No, consolidated attendance reports are maintained by individual colleges. Contact your college administration for attendance records."
            },
            {
                "question": "What is the difference between regular and Tatkal service for certificates?",
                "answer": "Regular services take longer to process (4-6 weeks), while Tatkal services are expedited and completed within 1-2 working days for an additional fee."
            },
            {
                "question": "Can I apply for duplicate certificates if my originals are stolen?",
                "answer": "Yes, file an FIR with the local police and submit it along with your application for duplicate certificates at the JNTUH Exam Branch."
            },
            {
                "question": "Are practical exams considered for revaluation?",
                "answer": "No, revaluation is only applicable to theory exam answer scripts. Practical exams and internal assessments are not eligible for revaluation."
            },
            {
                "question": "How can I find out if my course is AICTE approved?",
                "answer": "Check the JNTUH website or the AICTE approval list for your academic year to verify if your course and college are AICTE approved."
            },
            {
                "question": "What is the late fee for submitting exam fees after the deadline?",
                "answer": "The late fee for submitting exam fees is typically Rs. 1000. The exact fee depends on the duration of delay and is mentioned in the exam notification."
            },
            {
                "question": "How do I handle a discrepancy in marks uploaded online?",
                "answer": "Report discrepancies to your college examination cell immediately. They will coordinate with JNTUH to rectify the issue."
            },
            {
                "question": "Can I receive digital copies of my certificates via email?",
                "answer": "No, JNTUH does not provide digital copies via email. All certificates are issued as physical documents, though digital verification may be available."
            },
            {
                "question": "What is the procedure for attending improvement exams?",
                "answer": "Improvement exams are allowed for recently cleared subjects to improve grades. Apply through the JNTUH Exam Portal and pay the applicable fee per subject."
            },
                {
                    "question": "How can I obtain an equivalency certificate from JNTUH?",
                    "answer": "To obtain an equivalency certificate, submit an application through the Online Student Service Portal with your academic credentials, ID proof, and the required fee. The certificate will be issued after verification."
                },
                {
                    "question": "Can I write exams at a different JNTUH-affiliated college?",
                    "answer": "No, you are required to write your exams at the center allocated by JNTUH, which is usually based on your college and course registration."
                },
                {
                    "question": "What is the eligibility criteria for attending convocation?",
                    "answer": "To attend convocation, you must have completed your course and received your provisional certificate. Registration through the official portal is mandatory."
                },
                {
                    "question": "What happens if I miss the internal exams?",
                    "answer": "If you miss internal exams, you will not be able to retake them, as they are scheduled only once per semester. Your final internal marks will be calculated based on the exams you attended."
                },
                {
                    "question": "How do I retrieve my login credentials for the JNTUH portal?",
                    "answer": "If you forget your login credentials, use the 'Forgot Password' option on the JNTUH portal. Enter your registered email or mobile number to reset your password."
                },
                {
                    "question": "What is the process for grievance redressal regarding exam results?",
                    "answer": "Students can raise grievances about exam results through their college exam cell. The issue will be escalated to JNTUH for investigation and resolution."
                },
                {
                    "question": "Can I apply for revaluation if I passed but want better grades?",
                    "answer": "Yes, you can apply for revaluation even if you passed, provided the application is within the deadline. Revaluation is often used to improve grades."
                },
                {
                    "question": "What are the passing criteria for lab examinations?",
                    "answer": "To pass lab examinations, you must score a minimum of 50% in both internal and external evaluations combined."
                },
                {
                    "question": "Are there any restrictions on gap years for lateral entry students?",
                    "answer": "Lateral entry students must meet the eligibility criteria defined by JNTUH, which generally includes no restrictions on gap years but requires prior diploma completion."
                },
                {
                    "question": "What is the procedure for applying for a grace marks adjustment?",
                    "answer": "Grace marks can be applied for if you are failing by a small margin in one or two subjects. Submit a request through your college with supporting documentation and pay the prescribed fee."
                },
                {
                    "question": "How long does it take to process a duplicate degree certificate?",
                    "answer": "Processing a duplicate degree certificate typically takes 4-6 weeks under regular service. Tatkal requests can expedite this to 1-2 working days."
                },
                {
                    "question": "What is the role of the internal marks in final grading?",
                    "answer": "Internal marks contribute 25% to the final grade, with the remaining 75% coming from external examinations. Ensure consistent performance in internal assessments."
                },
                {
                    "question": "What should I do if my exam hall ticket has errors?",
                    "answer": "Immediately report any errors in your hall ticket to your college administration. They will coordinate with JNTUH to issue a corrected version."
                },
                {
                    "question": "How do I verify the authenticity of my degree certificate?",
                    "answer": "Employers or universities can verify the authenticity of your degree certificate through JNTUH’s official verification process, which may involve a formal request and payment of a verification fee."
                },
                {
                    "question": "What is the procedure to get a rank certificate from JNTUH?",
                    "answer": "Rank certificates are issued to top-performing students. Apply through the Online Student Service Portal with proof of eligibility, such as marks memos or rank notifications."
                },
                {
                    "question": "How can I check the exam timetable for my semester?",
                    "answer": "The exam timetable is published on the JNTUH official website under the Notifications or Exams section. Regularly check for updates closer to the exam period."
                },
                {
                    "question": "What is the re-admission policy at JNTUH?",
                    "answer": "Re-admission is allowed for students who discontinue due to valid reasons. You must apply for re-admission within the stipulated time, pay the applicable fees, and meet the eligibility criteria."
                },
                {
                    "question": "Can I change my exam center after it has been allocated?",
                    "answer": "No, exam centers are allocated based on JNTUH policies and cannot be changed once assigned. Ensure to check your center details carefully on your hall ticket."
                },
                {
                    "question": "What are the penalties for malpractice during exams?",
                    "answer": "Penalties for malpractice include annulment of exam results, suspension from future exams, or even course termination in severe cases. Students are advised to strictly adhere to exam guidelines."
                },
                {
                    "question": "How do I apply for a semester break due to health issues?",
                    "answer": "Submit a leave application through your college with supporting medical documents. Approval from the Head of the Department (HOD) and JNTUH is required for a semester break."
                },
                {
                    "question": "What is the fee structure for obtaining an original degree certificate?",
                    "answer": "The fee for an Original Degree (OD) varies depending on the year of graduation. Early bird applications typically cost Rs. 600-800, while late applications may incur additional charges."
                },
                {
                    "question": "Can I enroll in MOOCs as part of my credits at JNTUH?",
                    "answer": "Yes, JNTUH allows students to enroll in approved MOOCs for credit transfer. Ensure the MOOC is from a recognized platform and approved by your department."
                },
                {
                    "question": "What is the policy for repeating a semester at JNTUH?",
                    "answer": "Students failing to meet credit requirements may repeat a semester. However, this requires re-registration for courses and additional tuition fees."
                },
                {
                    "question": "How is the CGPA calculated at JNTUH?",
                    "answer": "The CGPA is calculated as the weighted average of the grades obtained in all semesters. Each subject’s credits contribute proportionally to the final CGPA."
                },
                {
                    "question": "What is the process for applying for an industry internship approval?",
                    "answer": "Submit an internship proposal to your college with details of the industry, project, and duration. Once approved, it will be forwarded to JNTUH for final endorsement."
                },
                {
                    "question": "What happens if I fail the same subject multiple times?",
                    "answer": "If you fail a subject multiple times, you must continue attempting supplementary exams. In some cases, the curriculum revision may require you to take an equivalent course."
                },
                {
                    "question": "How do I apply for the credit exemption policy?",
                    "answer": "To apply for credit exemption, submit an application through your college with details of the courses to be exempted and supporting documents such as course syllabi and grades."
                },
                {
                    "question": "What is the maximum number of attempts allowed for a subject?",
                    "answer": "There is no specific limit on attempts for clearing a subject, but students must complete their course within the maximum allowable duration (typically twice the course duration)."
                },
                {
                    "question": "How do I submit project work for final-year evaluation?",
                    "answer": "Submit your project report through your department by the deadline. Ensure all documentation, including your guide’s approval and plagiarism check report, is complete."
                },
                {
                    "question": "Can I attend supplementary exams if I have an academic year gap?",
                    "answer": "Yes, students with academic gaps can attend supplementary exams, provided they have valid hall tickets and meet other eligibility criteria."
                },
                    {
                        "question": "What are the attendance requirements to appear for JNTUH exams?",
                        "answer": "Students must maintain a minimum of 75% attendance in all subjects to be eligible to appear for exams. A 10% relaxation is given for medical or valid reasons with prior approval."
                    },
                    {
                        "question": "How can I apply for transcripts from JNTUH?",
                        "answer": "To apply for transcripts, visit the JNTUH Online Student Services Portal, fill out the application form, upload required documents, and pay the fee. Transcripts will be sent via post or can be collected in person."
                    },
                    {
                        "question": "What is the late fee for semester registration at JNTUH?",
                        "answer": "Late registration fees typically range between Rs. 100 to Rs. 1000, depending on the delay duration. Check the latest notification for the exact fee details."
                    },
                    {
                        "question": "Can I change my specialization during my M.Tech at JNTUH?",
                        "answer": "Specialization changes are generally not allowed once admission is confirmed. Consult your department or JNTUH for exceptions or policies related to unique cases."
                    },
                    {
                        "question": "What is the process for obtaining a provisional degree certificate?",
                        "answer": "Submit an application through the JNTUH Online Student Services Portal after your final semester results are declared. Ensure all fee dues are cleared before applying."
                    },
                    {
                        "question": "What is the penalty for not clearing a subject within the course duration?",
                        "answer": "If you fail to clear a subject within the maximum allowable course duration, you must register for additional attempts, and the degree may be delayed."
                    },
                    {
                        "question": "Can students take part-time jobs while studying under JNTUH regulations?",
                        "answer": "While JNTUH doesn’t explicitly prohibit part-time jobs, students must ensure that their work doesn’t interfere with academic commitments, including attendance and exams."
                    },
                    {
                        "question": "What is the re-registration policy for failed subjects?",
                        "answer": "Students failing a subject can re-register for it in subsequent semesters by paying a re-registration fee. This allows you to improve internal marks as well."
                    },
                    {
                        "question": "Are there any special considerations for students with disabilities during exams?",
                        "answer": "Yes, JNTUH provides facilities like additional exam time, scribe assistance, and special seating arrangements for students with disabilities. Submit supporting documents to avail of these provisions."
                    },
                    {
                        "question": "What is the procedure for withdrawing from a course at JNTUH?",
                        "answer": "To withdraw from a course, submit a formal application to your department head, citing valid reasons. Approval from the college and JNTUH is mandatory."
                    },
                    {
                        "question": "How does JNTUH handle cases of academic plagiarism?",
                        "answer": "JNTUH enforces strict anti-plagiarism policies. Submissions exceeding the permissible plagiarism percentage may face rejection, grade penalties, or disqualification."
                    },
                    {
                        "question": "Can I transfer from another university to JNTUH?",
                        "answer": "Transfers from other universities to JNTUH are rare and subject to strict criteria, including compatibility of curricula and availability of seats. Contact the admissions office for guidance."
                    },
                    {
                        "question": "What is the process for verifying an older degree certificate from JNTUH?",
                        "answer": "Degree verification requests for older certificates can be submitted through the JNTUH Online Verification Portal. Include the certificate details and pay the verification fee."
                    },
                    {
                        "question": "How can I access previous years’ question papers for preparation?",
                        "answer": "Previous question papers are available on the JNTUH official website or through your college library. Some student forums may also provide access to these resources."
                    },
                    {
                        "question": "What happens if my project report is not submitted on time?",
                        "answer": "Late submission of project reports can lead to penalties, including grade reduction or delayed evaluation. Seek an extension if you anticipate delays."
                    },
                    {
                        "question": "Are there any penalties for late tuition fee payment?",
                        "answer": "Yes, late payment of tuition fees incurs penalties ranging from nominal fines to possible de-registration if unpaid for an extended period."
                    },
                    {
                        "question": "How is the award of medals or ranks determined at JNTUH?",
                        "answer": "Medals and ranks are awarded based on academic excellence, typically considering CGPA and performance across all semesters without backlogs."
                    },
                    {
                        "question": "What are the rules for conducting mini-projects in the curriculum?",
                        "answer": "Mini-projects must adhere to the guidelines issued by the department. They should be submitted within the specified timeframe and undergo evaluation by internal and external examiners."
                    },
                    {
                        "question": "Can I apply for multiple revaluation requests simultaneously?",
                        "answer": "Yes, you can apply for revaluation of multiple subjects simultaneously. Submit separate applications and pay the applicable fee for each subject."
                    },
                    {
                        "question": "What is the dress code for attending exams at JNTUH?",
                        "answer": "While there isn’t a strict dress code, students are advised to wear formal or decent attire. Avoid wearing items like caps or large accessories that may violate exam rules."
                    },
                    {
                        "question": "Are JNTUH semester results published online or offline?",
                        "answer": "Semester results are published online on the JNTUH official website. Students can access them using their hall ticket numbers."
                    },
                    {
                        "question": "What is the procedure for rectifying errors in my mark sheet?",
                        "answer": "Submit an application through your college to JNTUH, detailing the error and providing supporting documents. Corrections typically take a few weeks to process."
                    },
                    {
                        "question": "How do I apply for a migration certificate after graduation?",
                        "answer": "Migration certificates can be requested through the JNTUH Online Student Services Portal. Provide your degree details and pay the requisite fee."
                    },
                    {
                        "question": "What is the maximum duration allowed for completing an M.Tech program?",
                        "answer": "The maximum allowable duration for completing an M.Tech program is four years, including any supplementary attempts."
                    },
                    {
                        "question": "What is the evaluation process for PhD dissertations at JNTUH?",
                        "answer": "PhD dissertations are evaluated through a combination of internal review, external examination by subject experts, and a viva-voce defense."
                    },
                    {
                        "question": "Can I switch from a regular to a part-time program in JNTUH?",
                        "answer": "Switching from regular to part-time programs may be allowed under specific conditions, such as employment. Submit a formal request to your department for consideration."
                    },
                    {
                        "question": "How can I find the syllabus for my course at JNTUH?",
                        "answer": "The syllabus is available on the JNTUH official website under the academics section. Departments may also provide printed copies or PDFs."
                    },
                    {
                        "question": "What is the policy for lateral entry students in B.Tech programs?",
                        "answer": "Lateral entry students join directly in the second year of the B.Tech program. They must have completed a diploma in engineering or its equivalent and meet JNTUH’s eligibility criteria."
                    },
                    {
                        "question": "Are there any scholarships available for JNTUH students?",
                        "answer": "Yes, JNTUH students can avail themselves of state government scholarships like fee reimbursement and merit-based scholarships. Check the college or university notice board for updates."
                    },
                    {
                        "question": "What is the process for changing exam subject choices during registration?",
                        "answer": "Subject changes during exam registration can only be done during the specified correction window. Submit a request through your college within the deadline."
                    }
                     
]

faq_questions = [item['question'] for item in faq_data]
faq_answers = [item['answer'] for item in faq_data]

faq_embeddings = model.encode(faq_questions)

def get_best_faq_answer(user_question):
    user_embedding = model.encode([user_question])

    similarities = cosine_similarity(user_embedding, faq_embeddings)

    most_similar_idx = np.argmax(similarities)

    most_similar_question = faq_questions[most_similar_idx]
    answer = faq_answers[most_similar_idx]

    return most_similar_question, answer

def run_app():
    st.title("JNTUH FAQ Chatbot")

    st.write("Hello! I'm your FAQ chatbot. Ask me anything about JNTUH services.")

    user_input = st.text_input("You:", "")

    if user_input:
        question, answer = get_best_faq_answer(user_input)
        st.write(f"Bot: I found a similar question: '{question}'")
        st.write(f"Answer: {answer}")
    
    if st.button("Exit"):
        st.write("Goodbye!")

if __name__ == "__main__":
    run_app()
