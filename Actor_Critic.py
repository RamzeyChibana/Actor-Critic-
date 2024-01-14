import tensorflow as tf
import numpy as np 
from Models import ActorCritic



class Agent():
    def __init__(self,num_actions,units=[1024,512],learning_rate = 0.01,gamma = 0.99):
        super().__init__()

        self.num_actions = num_actions
        self.gamma = gamma
        self.action = None

        self.brain = ActorCritic(units,num_actions,name="Actor_Cretic")
        self.brain.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


    def choose_action(self,state):
        state = tf.convert_to_tensor([state],dtype=tf.float32)
        _ , probs = self.brain(state)

        action = np.random.choice(self.num_actions,p=np.array(probs).squeeze())
        self.action = action
        return action
    
    def train(self,state,reward,state_,done):
        state = tf.convert_to_tensor([state],dtype=tf.float32) # current state
        state_ = tf.convert_to_tensor([state_],dtype = tf.float32) # next state
        reward = tf.convert_to_tensor(reward,dtype = tf.float32)

        with tf.GradientTape() as tape :
            state_value, probs = self.brain(state)
            state_value_ , _ = self.brain(state_)

            state_value = tf.squeeze(state_value) 
            state_value_ = tf.squeeze(state_value_) 
            
          

            log_prob = tf.math.log(probs[0,self.action])

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value

            actor_loss = -log_prob*delta
            critic_loss = delta**2

            total_loss = actor_loss+critic_loss
        gradient = tape.gradient(total_loss,self.brain.trainable_variables)
        self.brain.optimizer.apply_gradients(zip(gradient,self.brain.trainable_variables))






