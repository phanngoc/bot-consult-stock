<template>
  <div class="chat-container">
    <div class="chat-messages">
      <div v-for="(message, index) in messages" :key="index" 
        :class="{ 'user-message': message.isUser, 'bot-message': !message.isUser, 'message': true }">
        <div class="inner-message">
          <div class="text-message">
            {{ message.message }}
          </div>
          <div class="chart" v-if="!message.isUser && message.isChart">
            <Line
              :id="'line-id-' + index"
              :height="300"
              :width="600"
              :options="chartOptions"
              :data="dataChart(message)"
            />
          </div>
        </div>
      </div>
      <div id="loading-bar-spinner" class="spinner" v-if="isLoading">
        <div class="load-3">
          <div class="line"></div>
          <div class="line"></div>
          <div class="line"></div>
        </div>
      </div>

    </div>
    <div class="chat-input">
      <input v-model="userInput" @keyup.enter="sendMessage" class="input-box" />
    </div>
  </div>
</template>

<script>

import { Line } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale, PointElement,
  LineElement } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, LineElement, BarElement, CategoryScale, LinearScale, PointElement)

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
  components: {
    Line,
  },
  data() {
    return {
      isLoading: false,
      messages: [],
      userInput: '',
      chartData: {
        labels: ['January', 'February', 'March', 'April', 'May', 'June', 'July'],
        datasets: [
          {
            label: 'Data One',
            backgroundColor: '#f87979',
            data: [40, 39, 10, 40, 39, 80, 40]
          }
        ],
      },
      chartOptions: {
        responsive: false,
        maintainAspectRatio: false
      },
    };
  },
  computed: {

  },
  methods: {
    dataChart: function(message) {
      let stocks = message.data
      let datasets = []
      let labels = stocks[0].data.map(item => item.date)
      stocks.forEach(stockData => {
        let data = stockData.data.map(item => item.price)
        datasets.push({
          label: stockData.code,
          backgroundColor: '#f87979',
          data: data
        })
      });
      
      return {
        labels: labels,
        datasets: datasets,
      }
    },
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

@keyframes loadingC {
  0 {
    transform: translate(0, 0);
  }
  50% {
    transform: translate(0, 15px);
  }
  100% {
    transform: translate(0, 0);
  }
}

#loading-bar-spinner {
  display: flex;
  flex-direction: column;
  padding-bottom: 5px;
  .load-3{
    align-self: center;
  }
  .line {
    display: inline-block;
    width: 15px;
    height: 15px;
    border-radius: 15px;
    background-color: #4b9cdb;
  }

  .load-3 .line:nth-last-child(1) {
    animation: loadingC 0.6s 0.1s linear infinite;
  }
  .load-3 .line:nth-last-child(2) {
    animation: loadingC 0.6s 0.2s linear infinite;
  }
  .load-3 .line:nth-last-child(3) {
    animation: loadingC 0.6s 0.3s linear infinite;
  }
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
  --tw-bg-opacity: 1;
  background-color: rgba(247,247,248,var(--tw-bg-opacity));
  overflow-y: scroll;
  margin-bottom: 10px;
  flex: auto;
}

.message {
  background-color: #f0f0f0;
  padding: 10px;
  margin: 10px;
  border-radius: 8px;
  border-radius: 2px;
  .chart{
    display: block;
    width: 700px;
    height: 400px;
  }
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
    --tw-shadow: 0 0 15px rgba(0,0,0,.1);
    --tw-shadow-colored: 0 0 15px var(--tw-shadow-color);
    box-shadow: var(--tw-ring-offset-shadow,0 0 transparent),var(--tw-ring-shadow,0 0 transparent),var(--tw-shadow);
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
