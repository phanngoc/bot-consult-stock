<template>
  <div class="chat-container">
    <div class="chat-messages">
      <div v-for="(message, index) in messages" :key="index" 
        :class="{ 'user-message': message.isUser, 'bot-message': !message.isUser, 'message': true }">
        <div class="inner-message">
          <div class="text-message">
            {{ message.message }}
          </div>
        </div>
      </div>
      
      <div id="loading-bar-spinner" class="spinner" v-if="isLoading">
        <div class="spinner-icon"></div></div>

    </div>
    <div class="chat-input">
      <input v-model="userInput" @keyup.enter="sendMessage" class="input-box" />
    </div>
  </div>
</template>

<script>
import axios from 'axios';

const API_URL = 'http://127.0.0.1:5000/api/send-message';

export const getChatGPTResponse = async (userMessage) => {
  const response = await axios.post(API_URL, {
    prompt: userMessage,
  }, {
    headers: {
      'Content-Type': 'application/json;charset=UTF-8',
      "Access-Control-Allow-Origin": "*",
    },
  });
  console.log('getChatGPTResponse', response.data);
  return response.data;
};

export default {
  name: 'ChatComponent',
  props: {
    msg: String
  },
  data() {
    return {
      isLoading: false,
      messages: [],
      userInput: '',
    };
  },
  methods: {
    async sendMessage() {
      const userMessage = this.userInput;
      this.messages.push({ message: userMessage, isUser: true });
      this.userInput = '';
      this.isLoading = true;
      const botResponse = await getChatGPTResponse(userMessage);
      this.isLoading = false;
      if (botResponse.status == 1) {
        this.messages.push(botResponse);
      }
    },
  },
}
</script>

<style scoped>

#loading-bar-spinner.spinner {
    left: 50%;
    position: absolute;
    z-index: 19 !important;
    animation: loading-bar-spinner 400ms linear infinite;
}

#loading-bar-spinner.spinner .spinner-icon {
    width: 40px;
    height: 40px;
    border:  solid 4px transparent;
    border-top-color:  #00C8B1 !important;
    border-left-color: #00C8B1 !important;
    border-radius: 50%;
}

@keyframes loading-bar-spinner {
  0%   { transform: rotate(0deg);   transform: rotate(0deg); }
  100% { transform: rotate(360deg); transform: rotate(360deg); }
}


.chat-container {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto
/* 
  width: 100%;
  height: 100%; */
}
.chat-messages {
  margin-bottom: 10px;
  flex: auto;
}

.message {
  background-color: #f0f0f0;
  padding: 10px;
  margin: 10px;
  border-radius: 8px;
  border-radius: 2px;
}
.user-message {
  text-align: right;
  
}
.bot-message {
  text-align: left;
}

.chat-input {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  flex: none;
  .input-box {
    height: 60px;
    /* padding: 10px; */
    border: 1px solid #ccc;
    border-radius: 8px;
    width: 90%;
    font-size: 18px;
    font-weight: 600;
  }
}

</style>
