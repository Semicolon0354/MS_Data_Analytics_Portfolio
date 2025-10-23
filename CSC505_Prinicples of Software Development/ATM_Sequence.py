"""
Title: ATM_Sequence.py
@author: jdhum
"""
Comm_line_dict={0:'User to ATM',1:'ATM to Server',2:'Server to Bank',3:'Bank to Server',\
                4:'Server to ATM',5:'ATM to User'}

Insert_Card=['insert card','verify card',None,None,None,None]
Card_OK=[None,None,None,None,'Card OK','Request PIN']
Card_Reject=[None,None,None,None,'Card Invalid','Eject Card']
Enter_PIN=['enter PIN',['Verify PIN','PIN_Count=1'],None,None,None,None]
Correct_PIN=[None,None,None,None,'Correct PIN','Request Selection']
Incorrect_PIN_PIN_Count_less_3=[None,None,None,None,'incorrect PIN','Request PIN reentry']
Reenter_PIN=['Reenter PIN',['verify PIN','PIN_Count=PIN_Count+1'],None,None,None,None]
Incorrect_PIN_PIN_Count_is_3=[None,None,None,None,'incorrect PIN','eject card']
Balance_Check=['check balance','check balance','check balance','balance','balance','balance']
Withdrawal_Request=['amount to withdraw','amount to withdraw','check balance',None,None,None]
Withdrawal_Accepted=[None,None,None,'withdrawal accepted','withdrawal accepted','dispense cash']
Balance_Zero=[None,None,None,'account closed','account closed','account closed']
Insufficient_Funds=[None,None,None,'insufficient funds','insufficient funds','Eject Card']

Normal_operation=[Insert_Card,Card_OK,Enter_PIN,Correct_PIN,Balance_Check,\
                  Withdrawal_Request,Withdrawal_Accepted]
Card_Rejected=[Insert_Card,Card_Reject]

Incorrect_Pin=[Insert_Card,Card_OK,Enter_PIN,Incorrect_PIN_PIN_Count_less_3,\
               Reenter_PIN,Incorrect_PIN_PIN_Count_is_3]
    
Insufficient_Funds_Reject=[Insert_Card,Card_OK,Enter_PIN,Correct_PIN,Balance_Check,\
                  Withdrawal_Request, Insufficient_Funds]

Withdraw_to_Close=[Insert_Card,Card_OK,Enter_PIN,Correct_PIN,Balance_Check,\
                  Withdrawal_Request,Withdrawal_Accepted,Balance_Zero]

operation_select_dict={'A':Normal_operation,'B':Card_Rejected,'C':Incorrect_Pin,\
                       'D':Insufficient_Funds_Reject,'E':Withdraw_to_Close}
    
def step_output(output_list):
    for i in range(len(output_list)):
        if output_list[i] !=None:
            if type(output_list[i])==str:
                print(Comm_line_dict[i]+': '+output_list[i])
            else:
                print(Comm_line_dict[i]+':'+output_list[i][0])
                print(Comm_line_dict[i]+':'+output_list[i][1])
                
def output_biglist(big_list):
    for i in range(len(big_list)):
        step_output(big_list[i])
bool=True

while bool==True:        
    operation_steps=input("Enter the letter of the operation path you would like to\
see the steps for:\n\
A. Normal Operation\n\
B. Card Rejected\n\
C. Incorrect PIN\n\
D. Insufficient Funds\n\
E. Final Withdrawal\n")

    operation_steps=operation_steps.upper()
    output_biglist(operation_select_dict[operation_steps])
    bool2=True
    while bool2==True:
        repeat=input("\n\nWould you like to see another sequence? y/n")
        repeat=repeat.upper()
        if repeat=='Y':
            bool=True
            bool2=False
        elif repeat=='N':
            bool=False
            bool2=False
        else:
            print("Sorry, didn't catch that.")
        



