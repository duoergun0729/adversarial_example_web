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
        default: '150px'
      },
      chartData: {
        type: Array
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
          series: [
            {
              name: 'MNIST',
              type: 'pie',
              radius:['60%', '85%'],
              avoidLabelOverlap: false,
              label: {
                normal: {
                  show: false,
                  position: 'center'
                },
                emphasis: {
                  show: true,
                  textStyle: {
                    fontSize: '25',
                    fontWeight: 'bold'
                  }
                }
              },
              labelLine: {
                normal: {
                  show: false
                }
              },
              data: data,
              itemStyle: {
                normal: {
                  color: function (value) {
                    let j = -1;
                    const colorList = [
                      '#2ec7c9', '#b6a2de', '#5ab1ef', '#ffb980', '#d87a80',
                      '#8d98b3', '#e5cf0d', '#97b552', '#95706d', '#dc69aa'];
                    // const nameList = [
                    //   '0', '1', '2', '3', '4',
                    //   '5', '6', '7', '8', '9'];
                    const nameList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
                    for (let _i = 0; _i < 10; _i++) {
                      // console.log(value.data.name)
                      if (value.data.name[0] === nameList[_i])
                        j = _i
                    }
                    return colorList[j];
                  },
                }
              },
              animationEasing: 'cubicInOut',
              animationDuration: 2600
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
