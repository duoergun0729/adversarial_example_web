<template>
  <div class="updown-container">
    <el-row type="flex" justify="center">
      <el-col :span="12">
        <el-card class="box-card" style="margin-right: 0px; height: 500px">
          <div slot="header" class="clearfix">
            <svg-icon icon-class="form"/>
            <span style='margin-left:10px;'>上传图片</span>
          </div>
          <div>
            <el-upload
              class="avatar-uploader"
              action="/api_cifar/upload_cifar"
              :auto-upload='true'
              :show-file-list="false"
              :on-success="handleAvatarSuccess"
              :before-upload="beforeAvatarUpload">
              <img v-if="imageUrl" :src="imageUrl" class="avatar">
              <i v-else class="el-icon-plus avatar-uploader-icon"></i>
            </el-upload>
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
            <el-table :data="paramsData">
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


  // function convertImgToBase64(url, callback, outputFormat) {
  //   let canvas = document.createElement('CANVAS'),
  //     ctx = canvas.getContext('2d'),
  //     img = new Image;
  //   img.crossOrigin = 'Anonymous';
  //   img.onload = function () {
  //     canvas.height = img.height;
  //     canvas.width = img.width;
  //     ctx.drawImage(img, 0, 0);
  //     let dataURL = canvas.toDataURL(outputFormat || 'image/png');
  //     callback.call(this, dataURL);
  //     canvas = null;
  //   };
  //   img.src = url;
  // }

  let img_src = '';


  export default {
    name: 'cifar',
    data() {
      return {
        restaurants: '',
        loading: false,
        imageUrl: '',
        isShow: false,
        options: [{
          value: '-1',
          label: '不选择'
        }, {
          value: '0',
          label: '飞机'
        }, {
          value: '1',
          label: '汽车'
        }, {
          value: '2',
          label: '小鸟'
        }, {
          value: '3',
          label: '小猫'
        }, {
          value: '4',
          label: '小鹿'
        }, {
          value: '5',
          label: '小狗'
        }, {
          value: '6',
          label: '青蛙'
        }, {
          value: '7',
          label: '小马'
        }, {
          value: '8',
          label: '轮船'
        }, {
          value: '9',
          label: '卡车'
        }],
        select_value: '-1',
        activeName: 'upload',
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
          disturb: '0.03',
          target: '',
        }, {
          name: 'PGD',
          disturb: '0.03',
          target: '',
        }, {
          name: 'BIM',
          disturb: '0.03',
          target: '',
        }, {
          name: 'MIM',
          disturb: '0.03',
          target: '',
        }, {
          name: 'SMIM',
          disturb: '0.03',
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
      handleAvatarSuccess(res, file) {
        this.imageUrl = URL.createObjectURL(file.raw);
      },
      beforeAvatarUpload(file) {
        const isJPG = file.type === 'image/jpeg';
        const isPNG = file.type === 'image/png';
        const isLt2M = file.size / 1024 / 1024 < 2;
        //
        if (!isJPG && !isPNG) {
          this.$notify.error({
            title: '错误',
            message: '上传图片只能是 PNG/JPG 格式！'
          });
        }
        if (!isLt2M) {
          this.$notify.error({
            title: '错误',
            message: '上传图片大小不能超过 2MB!'
          });
        }
        return (isJPG || isPNG) && isLt2M;
      },
      checkBackend: function () {
        let self = this;
        axios.post('/api_cifar/check',
          Qs.stringify({})
        )
          .then(function (response) {
            let list = response.data;
            console.log(list.check)
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
        this.imageUrl = ''
        // this.$refs.upload.clearFiles()
      },
      drawInput: function () {
        let self = this
        let tdata = []
        if (this.imageUrl === '') {
          this.$notify.error({
            title: '错误',
            message: '必须先上传一张图片!'
          });
        }
        else {
          this.loading = true;
          console.log(self.paramsData)
          axios.post('/api_cifar/drawinput_cifar',
            Qs.stringify({
              // target: self.select_value,
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
              self.loading = false;
            })
        }
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
          {"label": "-1", "value": "不选择"},
          {"label": "1", "value": "飞机"},
          {"label": "2", "value": "汽车"},
          {"label": "3", "value": "小鸟"},
          {"label": "4", "value": "小猫"},
          {"label": "5", "value": "小鹿"},
          {"label": "6", "value": "小狗"},
          {"label": "7", "value": "青蛙"},
          {"label": "8", "value": "小马"},
          {"label": "9", "value": "轮船"},
          {"label": "10", "value": "卡车"},
        ]
      }
    },
    mounted() {
      this.clear();
      this.checkBackend();
      this.restaurants = this.loadAll();
    },

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

  .avatar-uploader {
    border: 2px dashed #d9d9d9;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
  }

  .avatar-uploader:hover {
    border-color: #409EFF;
  }

  .avatar-uploader-icon {
    font-size: 40px;
    color: #8c939d;
    width: 400px;
    height: 400px;
    line-height: 440px;
    text-align: center;
  }

  .avatar {
    width: 400px;
    height: 400px;
    display: block;
  }

  .box-card {
    /*width: 600px;*/
    margin: 20px;
  }

  body {
    margin: 0;
  }
</style>
