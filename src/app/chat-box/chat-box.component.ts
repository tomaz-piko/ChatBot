import { Component, OnInit, ViewChild } from '@angular/core';
import { NgbModule, NgbModal, NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';
import data from '../../assets/Models/SLO/data.json';
import dict from '../../assets/Models/SLO/dictionary.json';
import revDict from '../../assets/Models/SLO/revDictionary.json';
import * as tf from '@tensorflow/tfjs';
import { fetch as fetchPolyfill } from 'whatwg-fetch';
import { ModalService } from '../_modal';

@Component({
  selector: 'app-chat-box',
  templateUrl: './chat-box.component.html',
  styleUrls: ['./chat-box.component.css']
})
export class ChatBoxComponent implements OnInit {
  chat: any;
  message: any;
  encoder: any;
  decoder: any;
  title = 'About slo chatbot';
  data: any;
  @ViewChild("chatbox") MyEl: any;

  constructor(private modalService: ModalService) {
    this.chat = [];
    this.message = '';
  }

  ngOnInit(): void {
    this.loadModel();
    this.data = data;
  }

  scrollToBottom() {
    this.MyEl.nativeElement.scrollTop = this.MyEl.nativeElement.scrollHeight;
  }

  openModal(id: string): void {
    this.modalService.open(id);
  }

  closeModal(id: string): void {
    this.modalService.close(id);
  }

  async loadModel() {
    window.fetch = fetchPolyfill;
    this.encoder = await tf.loadLayersModel('assets/Models/SLO/Encoder_tf/model.json');
    console.log('Encoder loaded...');
    this.decoder = await tf.loadLayersModel('assets/Models/SLO/Decoder_tf/model.json');
    console.log('Decoder loaded...');
  }

  sendMessage() {
    if(this.message == '') {
      return; //Prazno sporočilo => se ne zgodi nič.
    }
    this.chat.push({user: 'me', msg: this.message});

    //Tensorflow odgovor
    let sequence = this.msgToSequence(this.message);
    let decoderData = this.seqToDecoderData(sequence);
    let reply = this.botReply(decoderData);
    //reply = this.formatMessage(reply);
    this.chat.push({user: 'bot', msg: reply});
    this.message = '';
  }

  seqToDecoderData(seq) {
    let input_seq = tf.buffer([1, data['enc_max_length']-2], 'int32');
    let index = 0;
    if(seq.length > data['enc_max_length']-2) {
      console.error('Sequence is to long.');
    }
    for (let i of seq) {
      input_seq.set(i, 0, index);
      index++;
    }
    return input_seq.toTensor();
  }

  botReply(input_seq) {
    let states_value = this.encoder.predict(input_seq) as tf.Tensor[];
    let target_seq = tf.buffer<tf.Rank.R2>([1, 1]);
    target_seq.set(dict['start'], 0, 0);
    let stop_condition = false;
    let decoded_sentence = '';
    let word_count = 1;
    while (!stop_condition) {
      const [output_tokens, h, c] = this.decoder.predict(
        [target_seq.toTensor(), ...states_value]
      ) as [
        tf.Tensor<tf.Rank.R2>,
        tf.Tensor<tf.Rank.R2>,
        tf.Tensor<tf.Rank.R2>,
      ];
      const sampledTokenIndex =
        output_tokens.squeeze().argMax(-1).arraySync() as number;
      let sampledWord = revDict[sampledTokenIndex];
      if (sampledWord != 'start' && sampledWord != 'end') {
        decoded_sentence += sampledWord + ' ';
        word_count++;
      }
      if (sampledWord == 'end' || word_count > data['dec_max_length']) {
        stop_condition = true;
      }
      target_seq = tf.buffer<tf.Rank.R2>([1, 1]);
      target_seq.set(sampledTokenIndex, 0, 0);
      states_value = [h, c]
    }
    return decoded_sentence
  }

  msgToSequence(msg : String) {
    let sequence = [];
    let words = this.message.toLowerCase().match(/\w+/g);
    let index = 0;
    for (let word of words) {
      if(typeof dict[word] != 'undefined') {
        sequence.push(dict[word]);
      }
      else {
        sequence.push(0)
      }
      index++;
    }
    return sequence;
  }

  formatMessage(msg : String) {
    let newMsg = '';
    newMsg += msg[0].toUpperCase();
    for(let i = 1; i < msg.length - 1; i++) {
        newMsg += msg[i];
    }
    newMsg += '.';
    return newMsg;
  }

  checkIfMe(user: string) {
    if(user == 'me') {
      return true;
    }
    else {
      return false;
    }
  }
}
