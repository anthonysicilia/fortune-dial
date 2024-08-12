import collections
import hashlib

from convokit import Corpus, download

from src.corpora.utils import natsort, normalize_instances
from src.corpora.utils import dump as dump_util

class ConvoKitFormatter:

    def __init__(self, name, seed=0, max_len=3600, write_situation=False):

        corpus = Corpus(filename=download(name))
        formatted_instances = collections.defaultdict(list)

        for convo in corpus.iter_conversations():
            utts = []
            chronology = self.get_chronology(convo)
            speakers = list()
            for utt in self.drop_label(chronology):
                speakers.append(utt.speaker.id)
                utts.append(self.format_utt(utt.text))
            norm = list(sorted(set(speakers))) # speaker order is important for hashing 
            utts = [f'Speaker {norm.index(speaker)}: {utt}'
                for utt, speaker in zip(utts, speakers)]
            inputs = ' || '.join(utts)
            # deal with very long instances:
            if max_len is not None and len(inputs) > max_len:
                inputs = '... || ' + inputs[-max_len:]
            order = self.get_internal_speaker_order(convo, speakers, norm)
            output = self.format_output(convo, order)
            situation = self.situation(convo, order)
            if write_situation and any(situation):
                s = ' || '.join([s for s in situation if s is not None])
                inputs = f'{s} || {inputs}'
            formatted_instances[convo.retrieve_meta('split')].append({'input': inputs, 'target': output,
                'situation' : situation, 'instance_id': hashlib.md5(inputs.encode('utf-8')).hexdigest()})
        
        self.formatted_instances = normalize_instances(formatted_instances, seed)
        self.seed = seed
    
    def drop_label(self, utts):
        raise NotImplementedError('Abstract class has unimplemented method: drop_label')

    def format_utt(self, utt):
        raise NotImplementedError('Abstract class has unimplemented method: format_utt')
    
    def format_output(self, convo):
        raise NotImplementedError('Abstract class has unimplemented method: format_output')
    
    def get_internal_speaker_id(self, utt):
        raise NotImplementedError('Abstract class has unimplemented method: get_internal_speaker_id')
    
    def situation(self, convo, order):
        return None
    
    def get_internal_speaker_order(self, convo, speakers, norm):
        normed_speakers = [norm.index(speaker) for speaker in speakers]
        interal_speakers = []
        for utt in self.drop_label(self.get_chronology(convo)):
            try:
                isid = self.get_internal_speaker_id(utt)
            except NotImplementedError:
                return None
            interal_speakers.append(isid)
        binding = {norm : internal for norm, internal in 
            zip(normed_speakers, interal_speakers)}
        return [binding[k] for k in sorted(binding.keys())]
    
    def get_custom_chronology(self, convo):
        # bad convo timestamps, use id to sort
        ids = convo.get_utterance_ids()
        chronology = []
        for i in natsort(ids):
            chronology.append(convo.get_utterance(i))
        return chronology
    
    def get_chronology(self, convo):
        try:
            return convo.get_chronological_utterance_list()
        except ValueError:
            return self.get_custom_chronology(convo)

    def dump(self, dname, fname):
        dump_util(self.formatted_instances, dname, fname, self.seed)
