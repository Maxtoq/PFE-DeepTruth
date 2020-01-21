from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import Adam


def get_3dConv_model():
    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation="relu",name="conv1", 
                     input_shape=(10,150,150,3),
                     strides=(1, 1, 1), padding="same"))  
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

    # 2nd layer group  
    model.add(Conv3D(128, (3, 3, 3), activation="relu",name="conv2", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

    # 3rd layer group   
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3a", 
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3b", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool3", padding="valid"))

    # 4th layer group  
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv4a", 
                     strides=(1, 1, 1), padding="same"))   
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv4b", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool4", padding="valid"))

    # 5th layer group  
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5a", 
                     strides=(1, 1, 1), padding="same"))   
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5b",
                      strides=(1, 1, 1), padding="same"))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool5", padding="valid"))
    model.add(Flatten())
                     
    # FC layers group
    model.add(Dense(2, activation='softmax', name='fc8'))

    model.compile(
        optimizer=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model