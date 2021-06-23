'''
    DataStructure to serialize a Keras Model
    It puts the network architecture as metadata and pickles the weights.
'''
import pickle
from soil.data_structures.data_structure import DataStructure


class Model(DataStructure):
    # @staticmethod
    @classmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return Model(pickle.loads(serialized), metadata)

    def serialize(self):
        ''' Function to serialize '''
        return pickle.dumps(self.data)


class SKLearnDataStructure(Model):
    ''' Data Structure for a Sklearn Model '''

    def get_data(self, **_args):
        # pylint: disable=no-self-use
        ''' Placeholder function for the API call '''
        return {"this is a model": True}  # self.data


class Model_DT(SKLearnDataStructure):
    # @classmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return Model_DT(pickle.loads(serialized), metadata)

    def get_data(self, **_args):
        return {"what i am?": "dt"} 


class Model_RF(SKLearnDataStructure):
    # @classmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return Model_RF(pickle.loads(serialized), metadata)

    def get_data(self, **_args):
        return {"what i am?": "RF"}


class MetaModel(Model):
    # @classmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return MetaModel(pickle.loads(serialized), metadata)

    # def serialize(self):
    #     for value in self.data:
    #         pickle.dump(value)

    def get_data(self, **_args):
        # pylint: disable=no-self-use
        ''' Placeholder function for the API call '''
        return {}  # self.data
