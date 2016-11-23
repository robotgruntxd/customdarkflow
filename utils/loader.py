"""
Here provides the interfaces to .ckpt and .weights files
"""

import tensorflow as tf
import os
import darknet as dn
import numpy as np

class loader(object):
    """
    interface to work with both .weights and .chkpt files
    in loading / recollecting / resolving mode
    """
    VAR_LAYER = ['convolutional', 'connected']

    def __init__(self, *args):
        self.setup(*args)
        self.base = -1

    def __call__(self, *key):
        key = list(key)
        for i in range(self.key_len):
            val = self.load(key[i:], i)
            if val is not None: return val
        return None
    
    def load(self, key, idx):
        b = self.base
        while b + 1 < len(self.src_key):
            b = b + 1; 
            key_b = self.src_key[b][idx:]
            if key_b == key:
                return self.yields(b)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

class weights_loader(loader):
    """one who understands .weights file"""

    # order of param flattened into .weights file
    _W_ORDER = dict({ 
        'convolutional': ['biases','scale','mean','var','kernel'],
        'connected': ['biases', 'weights']
    })

    def setup(self, path, src_layers):
        self.key_len = 1

        self.src_layers = src_layers
        walker = float32_walker(path, 16)
        self.src_key = [[l] for l in self.src_layers]

        self.vals = list()
        for i, layer in enumerate(src_layers):
            if walker.eof: 
                new = None
            else: 
                args = [i, layer.type]+layer.signature
                new = dn.darknet.create_darkop(*args)
            self.vals += [new]

            if new is None: continue
            if new.type not in self.VAR_LAYER: continue

            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.shape: continue
                val = walker.walk(new.size[par])
                new.w[par] = val
            new.finalize()

        if walker.path is not None:
            assert walker.offset == walker.size, \
            '{} failed: expect {} bytes, found {}'.format(
                path, walker.offset, walker.size)
            print 'Successfully identified {} bytes'.format(
                walker.size)

class checkpoint_loader(loader):
    """
    one who understands .ckpt files, very much
    """
    def setup(self, ckpt, ignore):
        self.key_len = 2

        meta = ckpt + '.meta'
        self.names = list()
        self.shape = list()
        self.vals = list()
        self.src_key = list()

        with tf.Graph().as_default() as graph:
            with tf.Session().as_default() as sess:
                saver = tf.train.import_meta_graph(meta)
                saver.restore(sess, ckpt)
                for var in tf.all_variables():
                    name = var.name.split(':')[0]
                    packet = [name, var.get_shape()]
                    self.src_key += [packet]
                    self.vals += [var.eval(sess)]

def create_loader(path, cfg = None):
    if path is None:
        load_type = weights_loader
    elif 'weights' in path:
        load_type = weights_loader
    else: 
        load_type = checkpoint_loader
    
    return load_type(path, cfg)

class float32_walker(object):
    """
    an incremental reader of float32 binary files.
    """
    def __init__(self, path, offset):
        self.eof = False # end of file
        self.offset = offset # current pos
        self.path = path # save the path
        if path is None: self.eof = True
        else: self.size = os.path.getsize(path)

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, \
        'Over read {}'.format(self.path)

        float32_1D_array = np.memmap(
            self.path, shape=(), mode='r', 
            offset = self.offset,
            dtype='({})float32,'.format(size)
        )

        self.offset = end_point
        if end_point == self.size: self.eof = True
        return float32_1D_array

def model_name(file_path):
    file_name = file_path.split('/')[-1]
    ext = str()
    if '.' in file_name: # exclude extension
        file_name = file_name.split('.')
        ext = file_name[-1]
        file_name = '.'.join(file_name[:-1])
    if ext == str() or ext == 'meta': # ckpt file
        file_name = file_name.split('-')
        num = int(file_name[-1])
        return '-'.join(file_name[:-1])
    if ext == 'weights':
        return file_name