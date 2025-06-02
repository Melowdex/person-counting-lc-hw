import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Conv2D, DepthwiseConv2D, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class ModelBuilder():

    def build_model_depth(input_shape: tuple, weights: str, alpha: float,
                    num_classes: int) -> tf.keras.Model:
        """ Construct a constrained object detection model.

        Args:
            input_shape: Passed to MobileNet construction.
            weights: Weights for initialization of MobileNet where None implies
                random initialization.
            alpha: MobileNet alpha value.
            num_classes: Number of classes, i.e. final dimension size, in output.

        Returns:
            Uncompiled keras model.

        Model takes (B, H, W, C) input and
        returns (B, H//8, W//8, num_classes) logits.
        """

        #! First create full mobile_net_V2 from (HW, HW, C) input
        #! to (HW/8, HW/8, C) output
        mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                    weights=weights,
                                    alpha=alpha,
                                    include_top=True)
        #! Default batch norm is configured for huge networks, let's speed it up
        for layer in mobile_net_v2.layers:
            if type(layer) == BatchNormalization:
                layer.momentum = 0.9
        #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
        cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
        #! Now attach a small additional head on the MobileNet
        model = Conv2D(filters=32, kernel_size=1, strides=1,
                    activation='relu', name='head')(cut_point.output)
        logits = Conv2D(filters=num_classes, kernel_size=1, strides=1,
                        activation=None, name='logits')(model)
        
        #Add layer for single output -> sum the output grid values 
        H, W, c = input_shape
        summed = DepthwiseConv2D(kernel_size=(int(H/8), int(W/8)),  
                                depth_multiplier=1,
                                strides=1,
                                padding='valid',
                                use_bias=False,
                                name='global_sum')(logits)

        # Reshape from (B, 1, 1, 1) → (B, 1)
        output = Reshape((1,), name='count_output')(summed)

        return Model(inputs=mobile_net_v2.input, outputs=output)
    

    def build_model_gap(input_shape: tuple, weights: str, alpha: float,
                    num_classes: int) -> tf.keras.Model:
        """ Construct a constrained object detection model.

        Args:
            input_shape: Passed to MobileNet construction.
            weights: Weights for initialization of MobileNet where None implies
                random initialization.
            alpha: MobileNet alpha value.
            num_classes: Number of classes, i.e. final dimension size, in output.

        Returns:
            Uncompiled keras model.

        Model takes (B, H, W, C) input and
        returns (B, H//8, W//8, num_classes) logits.
        """

        #! First create full mobile_net_V2 from (HW, HW, C) input
        #! to (HW/8, HW/8, C) output
        mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                    weights=weights,
                                    alpha=alpha,
                                    include_top=True)
        #! Default batch norm is configured for huge networks, let's speed it up
        for layer in mobile_net_v2.layers:
            if type(layer) == BatchNormalization:
                layer.momentum = 0.9
        #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
        cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
        #! Now attach a small additional head on the MobileNet
        model = Conv2D(filters=32, kernel_size=1, strides=1,
                    activation='relu', name='head')(cut_point.output)
        logits = Conv2D(filters=num_classes, kernel_size=1, strides=1,
                        activation=None, name='logits')(model)

        #Add layer for single output -> sum the output grid values 
        output = GlobalAveragePooling2D()(logits)

        return Model(inputs=mobile_net_v2.input, outputs=output)


    def build_model_dense(input_shape: tuple, weights: str, alpha: float,
                    num_classes: int) -> tf.keras.Model:
        """ Construct a constrained object detection model.

        Args:
            input_shape: Passed to MobileNet construction.
            weights: Weights for initialization of MobileNet where None implies
                random initialization.
            alpha: MobileNet alpha value.
            num_classes: Number of classes, i.e. final dimension size, in output.

        Returns:
            Uncompiled keras model.

        Model takes (B, H, W, C) input and
        returns (B, H//8, W//8, num_classes) logits.
        """

        #! First create full mobile_net_V2 from (HW, HW, C) input
        #! to (HW/8, HW/8, C) output
        mobile_net_v2 = MobileNetV2(input_shape=input_shape,
                                    weights=weights,
                                    alpha=alpha,
                                    include_top=True)
        #! Default batch norm is configured for huge networks, let's speed it up
        for layer in mobile_net_v2.layers:
            if type(layer) == BatchNormalization:
                layer.momentum = 0.9
        #! Cut MobileNet where it hits 1/8th input resolution; i.e. (HW/8, HW/8, C)
        cut_point = mobile_net_v2.get_layer('block_6_expand_relu')
        #! Now attach a small additional head on the MobileNet
        model = Conv2D(filters=32, kernel_size=1, strides=1,
                    activation='relu', name='head')(cut_point.output)
        logits = Conv2D(filters=num_classes, kernel_size=1, strides=1,
                        activation=None, name='logits')(model)

        #Add layer for single output -> sum the output grid values 
        d = tf.keras.layers.Dense(64, activation='relu')(logits)
        #tf.keras.layers.Dropout(0.3)(d)
        #tf.keras.layers.Dropout(0.3)(d)
        d = tf.keras.layers.Dense(32, activation='relu')(d)
        #tf.keras.layers.Dropout(0.2)(d)
        output = tf.keras.layers.Dense(1, activation='relu')(d)

        return Model(inputs=mobile_net_v2.input, outputs=output)    
