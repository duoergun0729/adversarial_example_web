<template>
  <div class="updown-container">
    <el-row type="flex" justify="center">
      <el-col :span="12">
        <el-card class="box-card" style="margin-right: 0px; height: 500px; width: 450px;">
          <div slot="header" class="clearfix">
            <svg-icon icon-class="form"/>
            <span style='margin-left:10px;'>数字画板</span>
          </div>
          <div>
            <canvas id='canvasScreen' ref="main"></canvas>
          </div>
        </el-card>
      </el-col>
      <el-col align="center">
        <el-card class="box-card" style="height: 500px;">
          <div slot="header" class="clearfix" align="left">
            <svg-icon icon-class="international"/>
            <span style='margin-left:10px;'>设置参数</span>
          </div>
          <div>
            <el-table :data="paramsData" @selection-change="handleSelectionChange" ref="multipleTable">
              <el-table-column
                type="selection"
                width="55">
              </el-table-column>
              <el-table-column class="big-size" prop="name" label="方法" min-width="12" align="center">
              </el-table-column>
              <el-table-column class="big-size" prop="disturb" label="扰动" min-width="22" align="center">
                <template slot-scope="scope">
                  <el-input-number class="input-width"
                                   v-model="scope.row.disturb"
                                   :step="0.01"
                                   :min="0"
                                   :max="10">
                  </el-input-number>
                </template>
              </el-table-column>
              <el-table-column class="big-size" prop="target" label="目标" min-width="22" align="center">
                <template slot-scope="scope">
                  <el-autocomplete
                    class="input-width"
                    v-model="scope.row.target"
                    :fetch-suggestions="querySearch"
                    placeholder="不选择"
                    @select="handleSelect"
                  >
                  </el-autocomplete>
                </template>
              </el-table-column>
            </el-table>
            <p>
              <el-button @click="clear" class='btn btn-default button-width'>清除</el-button>
              <el-button @click="drawInput" class='btn btn-default button-width'>上传</el-button>
            </p>
          </div>
        </el-card>
      </el-col>
    </el-row>
    <el-row>
      <el-card class="box-card" style="margin-top: 0px">
        <div slot="header" class="clearfix">
          <svg-icon icon-class="table"/>
          <span style='margin-left:10px;'>实验结果</span>
        </div>
        <div>
          <!--<div class="down-child">-->
          <el-table
            v-loading="loading"
            :data="tableData">
            <el-table-column type="expand">
              <template slot-scope="scope">
                <div class="container">
                  <div class="box">
                    <bar-chart :chart-data=scope.row.echarts></bar-chart>
                  </div>
                </div>
              </template>
            </el-table-column>
            <el-table-column class="big-size" prop="name" label="方法" min-width="22" align="center">
              <template slot-scope="scope">
                <div class="big-size" v-html="scope.row.name"></div>
              </template>
            </el-table-column>
            <el-table-column prop="img" label="图像" min-width="40" min-height="170" align="center">
              <template slot-scope="scope">
                <img :src="scope.row.img"/>
              </template>
            </el-table-column>
            <el-table-column prop="attack_result" label="结果" min-width="22" align="center">
              <template slot-scope="scope">
                <div class="big-size" v-html="scope.row.attack_result"></div>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </el-card>
    </el-row>
  </div>
</template>

<script>
  import axios from 'axios'
  import Qs from 'qs'
  import BarChart from './components/BarChart'


  export default {
    name: 'mnist',
    data() {
      return {
        multipleSelection: [],
        restaurants: '',
        loading: false,
        isShow: false,
        input_fgsm: '0.03',
        input_pgd: '0.03',
        input_bim: '0.03',
        input_mim: '0.03',
        input_smim: '0.03',
        target_fgsm: '-1',
        target_pgd: '-1',
        target_bim: '-1',
        target_mim: '-1',
        target_smim: '-1',
        echarts: [],
        // echarts_defense: [],


        state1: '',
        paramsData: [{
          name: 'FGSM',
          disturb: '0.3',
          target: '',
        }, {
          name: 'PGD',
          disturb: '0.3',
          target: '',
        }, {
          name: 'BIM',
          disturb: '0.3',
          target: '',
        }, {
          name: 'MIM',
          disturb: '0.3',
          target: '',
        }, {
          name: 'SMIM',
          disturb: '0.3',
          target: '',
        }],
        tableData: [{
          name: 'CLEAN',
          img: '',
          attack_result: '',
        }, {
          name: 'FGSM',
          img: '',
          attack_result: '',
        }, {
          name: 'PGD',
          img: '',
          attack_result: '',
        }, {
          name: 'BIM',
          img: '',
          attack_result: '',
        }, {
          name: 'MIM',
          img: '',
          attack_result: '',
        }, {
          name: 'SMIM',
          img: '',
          attack_result: '',
        }],
      }
    },
    components: {
      BarChart
    },
    methods: {
      checkBackend: function () {
        let self = this;
        axios.post('/api_mnist/check',
          Qs.stringify({})
        )
          .then(function (response) {
            let list = response.data;
            if (list.check === true) {
              self.$message({
                message: '成功连接到服务器',
                type: 'success'
              });
            }
          })
          .catch(function (error) {
            self.$message({
              message: '连接服务器失败！',
              type: 'error'
            });
          })
      },
      clear: function () {
        this.tableData = [{
          name: 'CLEAN',
          img: '',
          attack_result: '',
        }, {
          name: 'FGSM',
          img: '',
          attack_result: '',
        }, {
          name: 'PGD',
          img: '',
          attack_result: '',
        }, {
          name: 'BIM',
          img: '',
          attack_result: '',
        }, {
          name: 'MIM',
          img: '',
          attack_result: '',
        }, {
          name: 'SMIM',
          img: '',
          attack_result: '',
        }];
        this.ctx = this.$refs.main.getContext('2d')
        this.ctx.fillStyle = '#FFFFFF'
        this.ctx.fillRect(0, 0, 400, 400)
        this.ctx.lineWidth = 1
        this.ctx.strokeRect(0, 0, 400, 400)
        this.ctx.lineWidth = 0.05
      },
      handleSelectionChange(val) {
        this.multipleSelection = val;
        console.log(this.multipleSelection)
      },
      toggleSelection(rows) {
        if(!rows) {
          return ;
        }
        rows.forEach(row => {
          this.$refs.multipleTable.toggleRowSelection(row);
        });
      },
      drawInput: function () {
        let self = this
        let tdata = []
        let img = new Image()
        img.onload = function () {
          let inputs = []
          const small = document.createElement('canvas').getContext('2d')
          small.drawImage(img, 0, 0, img.width, img.height, 0, 0, 28, 28)
          let data = small.getImageData(0, 0, 28, 28).data
          for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
              let n = 4 * (i * 28 + j)
              inputs[i * 28 + j] = (data[n] + data[n + 1] + data[n + 2]) / 3
            }
          }
          if (Math.min.apply(Math, inputs) === 255) {
            return
          }
          axios.post('/api_mnist/drawinput_mnist',
            Qs.stringify({
              inputs: JSON.stringify(inputs),
              multipleSelection: self.multipleSelection,
              fgsm_disturb: self.paramsData[0].disturb,
              pgd_disturb: self.paramsData[1].disturb,
              bim_disturb: self.paramsData[2].disturb,
              mim_disturb: self.paramsData[3].disturb,
              smim_disturb: self.paramsData[4].disturb,
              fgsm_target: self.paramsData[0].target,
              pgd_target: self.paramsData[1].target,
              bim_target: self.paramsData[2].target,
              mim_target: self.paramsData[3].target,
              smim_target: self.paramsData[4].target,
            })
          )
            .then(function (response) {
              let list = response.data;
              for (let _i = 0; _i < response.data['name'].length; _i++) {
                let obj = {};
                obj.name = list.name[_i];
                obj.img = list.img[_i];
                obj.attack_result = list.attack_result[_i];
                // obj.defense_result = list.defense_result[_i];
                obj.echarts = list.echarts[_i];
                // obj.echarts_defense = list.echarts_defense[_i]
                tdata[_i] = obj;
              }
              self.tableData = tdata;
            })
        }
        img.src = this.$refs.main.toDataURL()
      },
      querySearch(queryString, cb) {
        var restaurants = this.restaurants;
        var results = queryString ? restaurants.filter(this.createFilter(queryString)) : restaurants;
        // 调用 callback 返回建议列表的数据
        cb(results);
        console.log(this.state1)
      },
      createFilter(queryString) {
        return (restaurant) => {
          return (restaurant.value.toLowerCase().indexOf(queryString.toLowerCase()) !== -1);
        };
      },
      handleSelect(row) {
        // console.log(row);
      },
      loadAll() {
        return [
          {"label": "1", "value": "1"},
          {"label": "2", "value": "2"},
          {"label": "3", "value": "3"},
          {"label": "4", "value": "4"},
          {"label": "5", "value": "5"},
          {"label": "6", "value": "6"},
          {"label": "7", "value": "7"},
          {"label": "8", "value": "8"},
          {"label": "9", "value": "9"},
          {"label": "10", "value": "0"},
        ]
      }
    }
    ,
    mounted() {
      this.toggleSelection(this.paramsData);
      this.checkBackend();
      (function () {
        let _createClass = function () {
          function defineProperties(target, props) {
            for (let i = 0; i < props.length; i++) {
              let descriptor = props[i];
              descriptor.enumerable = descriptor.enumerable || false;
              descriptor.configurable = true;
              if ("value" in descriptor) descriptor.writable = true;
              Object.defineProperty(target, descriptor.key, descriptor);
            }
          }

          return function (Constructor, protoProps, staticProps) {
            if (protoProps) defineProperties(Constructor.prototype, protoProps);
            if (staticProps) defineProperties(Constructor, staticProps);
            return Constructor;
          };
        }();

        function _classCallCheck(instance, Constructor) {
          if (!(instance instanceof Constructor)) {
            throw new TypeError("Cannot call a class as a function");
          }
        }

        const Main = function () {
          function Main() {
            _classCallCheck(this, Main)
            this.canvas = document.getElementById('canvasScreen')
            this.input = document.getElementById('input')
            this.update = document.getElementById('update')
            this.canvas.width = 400 // 16 * 28 + 1
            this.canvas.height = 400 // 16 * 28 + 1
            this.ctx = this.canvas.getContext('2d')
            this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this))
            this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this))
            this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this))
          }

          _createClass(Main, [
            {
              key: 'onMouseDown',
              value: function onMouseDown(e) {
                this.canvas.style.cursor = 'default'
                this.drawing = true
                this.prev = this.getPosition(e.clientX, e.clientY)
              }
            }, {
              key: 'onMouseUp',
              value: function onMouseUp() {
                this.drawing = false
                // this.drawInput()
              }
            }, {
              key: 'onMouseMove',
              value: function onMouseMove(e) {
                if (this.drawing) {
                  let curr = this.getPosition(e.clientX, e.clientY)
                  this.ctx.lineWidth = 25
                  this.ctx.lineCap = 'round'
                  this.ctx.beginPath()
                  this.ctx.moveTo(this.prev.x, this.prev.y)
                  this.ctx.lineTo(curr.x, curr.y)
                  this.ctx.stroke()
                  this.ctx.closePath()
                  this.prev = curr
                }
              }
            }, {
              key: 'getPosition',
              value: function getPosition(clientX, clientY) {
                let rect = this.canvas.getBoundingClientRect()
                return {
                  x: clientX - rect.left,
                  y: clientY - rect.top
                }
              }
            }])
          return Main
        }()
        new Main()
      })()
      this.clear()
      this.restaurants = this.loadAll();
    }

  }
</script>

<style scoped>
  .parent {
    display: flex;
  }

  .child1 {
    flex: 0.3;
    margin: 10px 30px;
  }

  .child2 {
    flex: 0.7;
    margin-top: 110px;
    margin-left: 20px;
  }

  .span-text {
    display: inline-block;
    width: 150px;
    text-align: right;
  }

  .input-width {
    width: 150px;
  }

  .button-width {
    width: 140px;
  }

  .box {
    width: 50%;
    float: left;
    display: inline;
    text-align: center;
  }

  .big-size {
    font-size: 18px;
  }

  .up-child {
    flex: 0.3;
  }

  .down-child {
    flex: 0.7;
    /*margin: 10px;*/

  }

  .updown-container {
    display: flex;
    flex-direction: column;
    /*background-color: rgb(240, 242, 245);*/
  }

  .box-card {
    /*width: 600px;*/
    margin: 20px;
  }

  body {
    margin: 0;
  }
</style>
