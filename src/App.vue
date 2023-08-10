<template>
  <div id="sidebar">
    <div class="item-list">
      <div class="item">
        <!-- Some chat history item -->
      </div>
    </div>
  </div>
  <ChatComponent />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</template>

<script>
import ChatComponent from './components/ChatComponent.vue'

export default {
  name: 'App',
  components: {
    ChatComponent
  }
}
</script>

<style>
html {
  display: block;
  width: 100%;
  height: 100%;
}
body{
  display: block;
  width: 100%;
  height: 100%;
}

#sidebar {
  width: 200px;
  /* height: 100%; */
  background-color: #6d9a80;
  padding: 10px;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: row;
}
</style>
