import tensorflow as tf 
import keras 
from keras import layers 
import os



class ActorCritic(tf.keras.Model):
    def __init__(self,units,num_actions,name,checkpoint_dir="models"):
        super().__init__()

        self.units = units
        self.num_actions = num_actions
        self.model_name = name
        self.checkpoint_dir = os.path.join(checkpoint_dir,self.model_name)


        self.commun1 = layers.Dense(self.units[0],activation="relu")
        self.commun2 = layers.Dense(self.units[1],activation="relu")
        self.critic = layers.Dense(1,name="critic") # Compute State Value function
        self.agent = layers.Dense(self.num_actions,name="agent",activation="softmax") # Policy Distrubtion


    def call(self,state):

        x = self.commun1(state)
        x = self.commun2(x)

        value = self.critic(x)
        pi = self.agent(x)

        return value,pi
    


if __name__ == "__main__":
    print("test the Model..")
    model = ActorCritic([1024,512],2,"test")
    out = model(tf.random.normal(shape=(1,5)))
    print(out)
 
    model.summary()





