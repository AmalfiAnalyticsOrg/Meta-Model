'''
    DataStructure to serialize a Keras Model
    It puts the network architecture as metadata and pickles the weights.
'''
import pickle
from soil import logger
from soil.data_structures.data_structure import DataStructure


class BaselineModel(DataStructure):
    ''' Data Structure for a Sklearn Model '''
    @staticmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return BaselineModel(pickle.loads(serialized), metadata)

    def serialize(self):
        ''' Function to serialize '''
        return pickle.dumps(self.data)

    def get_data(self, **_args):
        # pylint: disable=no-self-use
        ''' Placeholder function for the API call '''
        return {"HEEEEEEY": True}  # self.data Parametres del Model? 


class IDS(DataStructure):
    ''' Data Structure for a Metamodels ids '''
    @staticmethod
    def unserialize(serialized, metadata):
        ''' Function to deserialize '''
        return IDS(pickle.loads(serialized), metadata)

    def serialize(self):
        ''' Function to serialize '''
        return pickle.dumps(self.data)

    def get_data(self, **_args):
        # pylint: disable=no-self-use
        ''' Placeholder function for the API call '''
        logger.info('GET DATA, LIST')
        logger.info(list(self.data))
        return list(self.data)  # self.data Parametres del Model? 