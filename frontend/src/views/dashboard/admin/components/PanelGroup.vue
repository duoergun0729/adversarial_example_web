<template>
  <el-row class="panel-group" :gutter="40">
    <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
      <el-card class="box-card" style="margin-top: 40px">
        <div slot="header" class="clearfix">
          <svg-icon icon-class="form"/>
          <span style='margin-left:10px;'>MNIST</span>
        </div>
        <div align="center">
          <router-link class="pan-btn blue-btn" to="/mnist">MNIST</router-link>
        </div>
      </el-card>
    </el-col>
    <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
      <el-card class="box-card" style="margin-top: 40px">
        <div slot="header" class="clearfix">
          <svg-icon icon-class="example"/>
          <span style='margin-left:10px;'>CIFAR-10</span>
        </div>
        <div align="center">
          <router-link class="pan-btn blue-btn" to="/cifar">CIFAR-10</router-link>
        </div>
      </el-card>
    </el-col>
    <el-col :xs="16" :sm="16" :lg="8" class="card-panel-col">
      <el-card class="box-card" style="margin-top: 40px">
        <div slot="header" class="clearfix">
          <svg-icon icon-class="star"/>
          <span style='margin-left:10px;'>ImageNet</span>
        </div>
        <div align="center">
          <router-link class="pan-btn blue-btn" to="/imagenet">ImageNet</router-link>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script>
  import CountTo from 'vue-count-to'

  export default {
    components: {
      CountTo
    },
    methods: {
      handleSetLineChartData(type) {
        this.$emit('handleSetLineChartData', type)
      }
    }
  }
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
  .panel-group {
    margin-top: 18px;
    .card-panel-col {
      margin-bottom: 32px;
    }
    .card-panel {
      height: 108px;
      cursor: pointer;
      font-size: 12px;
      position: relative;
      overflow: hidden;
      color: #666;
      background: #fff;
      box-shadow: 4px 4px 40px rgba(0, 0, 0, .05);
      border-color: rgba(0, 0, 0, .05);
      &:hover {
        .card-panel-icon-wrapper {
          color: #fff;
        }
        .icon-people {
          background: #40c9c6;
        }
        .icon-message {
          background: #36a3f7;
        }
        .icon-money {
          background: #f4516c;
        }
        .icon-shoppingCard {
          background: #34bfa3
        }
      }
      .icon-people {
        color: #40c9c6;
      }
      .icon-message {
        color: #36a3f7;
      }
      .icon-money {
        color: #f4516c;
      }
      .icon-shoppingCard {
        color: #34bfa3
      }
      .card-panel-icon-wrapper {
        float: left;
        margin: 14px 0 0 14px;
        padding: 16px;
        transition: all 0.38s ease-out;
        border-radius: 6px;
      }
      .card-panel-icon {
        float: left;
        font-size: 48px;
      }
      .card-panel-description {
        float: right;
        font-weight: bold;
        margin: 26px;
        margin-left: 0px;
        .card-panel-text {
          line-height: 18px;
          color: rgba(0, 0, 0, 0.45);
          font-size: 16px;
          margin-bottom: 12px;
        }
        .card-panel-num {
          font-size: 20px;
        }
      }
    }
  }
</style>
