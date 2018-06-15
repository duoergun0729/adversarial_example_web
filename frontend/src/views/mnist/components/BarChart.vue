<template>
  <div :class="className" :style="{height:height,width:width}"></div>
</template>

<script>
  import echarts from 'echarts'

  require('echarts/theme/macarons') // echarts theme
  import {debounce} from '@/utils'

  export default {
    props: {
      className: {
        type: String,
        default: 'chart'
      },
      width: {
        type: String,
        default: '100%'
      },
      height: {
        type: String,
        default: '300px'
      },
      chartData: {
        type: Object
      }
    },
    data() {
      return {
        chart: null
      }
    },
    watch: {
      chartData: {
        deep: true,
        handler(val) {
          this.setOptions(val)
        }
      }
    },
    mounted() {
      this.initChart()
      this.__resizeHanlder = debounce(() => {
        if (this.chart) {
          this.chart.resize()
        }
      }, 100)
      window.addEventListener('resize', this.__resizeHanlder)
    },
    beforeDestroy() {
      if (!this.chart) {
        return
      }
      window.removeEventListener('resize', this.__resizeHanlder)
      this.chart.dispose()
      this.chart = null
    },
    methods: {
      setOptions(data = []) {
        // initChart() {
        //   this.chart = echarts.init(this.$el, 'macarons')

        this.chart.setOption({
          tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}({d}%)'
          },
          // legend: {
          //   left: 'center',
          //   bottom: '10',
          //   data: ['Industries', 'Technology', 'Forex', 'Gold', 'Forecasts']
          // },
          calculable: true,
          xAxis: {
            type: 'category',
            data: data['name'],
            axisLabel: {
              interval: 0,
              rotate: 0,//倾斜度 -90 至 90 默认为0
              margin: 2,
              // textStyle: {
              //   fontWeight: "bolder",
              //   color: "#000000"
              // }
            },
          },
          yAxis: {
            type: 'value',
            min: 0,
            max: 1,
          },
          series: [
            {
              name: 'MNIST',
              type: 'bar',
              data: data['value'],
              // animationEasing: 'cubicInOut',
              // animationDuration: 2600
            }
          ]
        })
      },
      initChart() {
        console.log(this.chartData)
        this.chart = echarts.init(this.$el, 'macarons')
        this.setOptions(this.chartData)
      }
    }
  }
</script>
