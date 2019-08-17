import tensorflow as tf
import numpy as np

class DQN():
    def __init__(self, K, scope, image_size):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder_with_default(False, (), 'is_training')
            self.X = tf.placeholder(tf.float32, shape=(None, image_size,image_size, 4), name='X')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            Z = self.X / 255.0
            #Z = tf.layers.batch_normalization(Z, training=self.is_training)
            Z = tf.layers.conv2d(Z, 32, [8,8], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            #Z = tf.layers.batch_normalization(Z, training=self.is_training)
            Z = tf.layers.conv2d(Z, 64, [4,4], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            #Z = tf.layers.batch_normalization(Z, training=self.is_training)
            Z = tf.layers.conv2d(Z, 64, [3,3], activation=tf.nn.relu)
            Z = tf.layers.max_pooling2d(Z,[2,2],2)
            Z = tf.contrib.layers.flatten(Z)
            #Z = tf.layers.batch_normalization(Z, training=self.is_training)
            Z = tf.layers.dense(Z, 512, activation=tf.nn.relu)

            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(5e-6).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        self.session = session
    
    def predict(self, states):
        return self.session.run(self.predict_op, feed_dict = {self.X: states, self.is_training: False})
    
    def update(self, states, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op, self.update_ops],
                feed_dict = {self.X: states, self.G: targets, self.actions: actions, self.is_training: True}
                )[0]
        return c
    
    def sample_action(self,x,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])





class DQN_multicamera():
    def __init__(self, K, scope, image_size1,image_size2):
        self.K = K
        self.scope = scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder_with_default(False, (), 'is_training')
            self.X1 = tf.placeholder(tf.float32, shape=(None, image_size1,image_size1, 4), name='X1')
            self.X2 = tf.placeholder(tf.float32, shape=(None, image_size2,image_size2, 4), name='X2')
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
            Z1 = self.X1 / 255.0
            #Z1 = tf.layers.batch_normalization(Z1, training=self.is_training)
            Z1 = tf.layers.conv2d(Z1, 32, [4,4], activation=tf.nn.relu)
            Z1 = tf.layers.max_pooling2d(Z1,[2,2],2)
            #Z1 = tf.layers.batch_normalization(Z1, training=self.is_training)
            Z1 = tf.layers.conv2d(Z1, 64, [4,4], activation=tf.nn.relu)
            Z1 = tf.layers.max_pooling2d(Z1,[2,2],2)
            #Z1 = tf.layers.batch_normalization(Z1, training=self.is_training)
            Z1 = tf.layers.conv2d(Z1, 128, [3,3], activation=tf.nn.relu)
            Z1 = tf.layers.max_pooling2d(Z1,[2,2],1)
            Z1 = tf.contrib.layers.flatten(Z1)
            Z1 = tf.layers.dense(Z1, 512, activation=tf.nn.relu)

            Z2 = self.X2 / 255.0
            #Z2 = tf.layers.batch_normalization(Z2, training=self.is_training)
            Z2 = tf.layers.conv2d(Z2, 32, [4,4], activation=tf.nn.relu)
            Z2 = tf.layers.max_pooling2d(Z2,[2,2],2)
            #Z2 = tf.layers.batch_normalization(Z2, training=self.is_training)
            Z2 = tf.layers.conv2d(Z2, 64, [4,4], activation=tf.nn.relu)
            Z2 = tf.layers.max_pooling2d(Z2,[2,2],2)
            #Z2 = tf.layers.batch_normalization(Z2, training=self.is_training)
            Z2 = tf.layers.conv2d(Z2, 128, [3,3], activation=tf.nn.relu)
            Z2 = tf.layers.max_pooling2d(Z2,[2,2],1)
            Z2 = tf.contrib.layers.flatten(Z2)
            Z2 = tf.layers.dense(Z2, 512, activation=tf.nn.relu)
            
            Z = tf.concat([Z1,Z2], axis = 1)
            #Z = tf.layers.batch_normalization(Z, training=self.is_training)
            Z = tf.layers.dense(Z, 1024, activation=tf.nn.relu)
            self.predict_op = tf.layers.dense(Z,K, activation=tf.nn.relu)
            selected_action_value = tf.reduce_sum(self.predict_op * tf.one_hot(self.actions,K), reduction_indices=[1])
            
            cost = tf.reduce_mean(tf.losses.huber_loss(self.G, selected_action_value))
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
            self.cost = cost
            
            
    def copy_from(self, other):
        mine = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        mine = sorted(mine, key=lambda v: v.name)
        theirs = [t for t in tf.trainable_variables() if t.name.startswith(other.scope)]
        theirs = sorted(theirs, key=lambda v: v.name)
        
        ops = []
        for p,q in zip(mine, theirs):
            op = p.assign(q)
            ops.append(op)
        self.session.run(ops)
    
    def save(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        params = self.session.run(params)
        np.savez('tf_dqn_weights.npz', *params)
    
    def load(self):
        params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        npz = np.load('tf_dqn_weights.npz')
        ops = []
        for p, (_, v) in zip(params, npz.iteritems()):
            ops.append(p.assign(v))
        self.session.run(ops)
        
    def set_session(self,session):
        self.session = session
    
    def predict(self, states1, states2):
        return self.session.run(self.predict_op, feed_dict = {self.X1: states1, self.X2: states2, self.is_training: False})
    
    def update(self, states1, states2, actions, targets):
        c = self.session.run(
                [self.cost, self.train_op, self.update_ops],
                feed_dict = {self.X1: states1, self.X2: states2, self.G: targets, self.actions: actions, self.is_training: True}
                )[0]
        return c
    
    def sample_action(self,states1, states2,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([states1], [states2])[0])