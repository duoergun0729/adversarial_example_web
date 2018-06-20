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
              action="/api_imagenet/upload_imagenet"
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
                  <!--<div class="box">-->
                  <!--<bar-chart :chart-data=scope.row.echarts_defense></bar-chart>-->
                  <!--</div>-->
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
            <el-table-column prop="img" label="扰动图像" min-width="40" min-height="170" align="center">
              <template slot-scope="scope">
                <img :src="scope.row.imgnoise"/>
              </template>
            </el-table-column>
            <el-table-column prop="attack_result" label="结果" min-width="22" align="center">
              <template slot-scope="scope">
                <div class="big-size" v-html="scope.row.attack_result"></div>
              </template>
            </el-table-column>
            <!--<el-table-column prop="defense_result" label="结果2" min-width="22" align="center">-->
            <!--<template slot-scope="props">-->
            <!--<div class="big-size" v-html="props.row.defense_result"></div>-->
            <!--</template>-->
            <!--</el-table-column>-->
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


  function convertImgToBase64(url, callback, outputFormat) {
    let canvas = document.createElement('CANVAS'),
      ctx = canvas.getContext('2d'),
      img = new Image;
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
      canvas.height = img.height;
      canvas.width = img.width;
      ctx.drawImage(img, 0, 0);
      let dataURL = canvas.toDataURL(outputFormat || 'image/png');
      callback.call(this, dataURL);
      canvas = null;
    };
    img.src = url;
  }

  let img_src = '';


  export default {
    name: 'imagenet',
    data() {
      return {
        restaurants: '',
        loading: false,
        imageUrl: '',
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
          imgnoise: '',
          attack_result: '',
        }, {
          name: 'FGSM',
          img: '',
          imgnoise: '',
          attack_result: '',
        }, {
          name: 'PGD',
          img: '',
          imgnoise: '',
          attack_result: '',
        }, {
          name: 'BIM',
          img: '',
          imgnoise: '',
          attack_result: '',
        }, {
          name: 'MIM',
          img: '',
          imgnoise: '',
          attack_result: '',
        }, {
          name: 'SMIM',
          img: '',
          imgnoise: '',
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
        axios.post('/api_imagenet/check',
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
          axios.post('/api_imagenet/drawinput_imagenet',
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
                obj.imgnoise = list.imgnoise[_i];
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
      loadAll() {
        return [
          {"label": "-1", "value": "不选择"},
          {"label": "1", "value": "丁鲷"},
          {"label": "2", "value": "金鱼"},
          {"label": "3", "value": "大白鲨"},
          {"label": "4", "value": "虎鲨"},
          {"label": "5", "value": "锤头鲨"},
          {"label": "6", "value": "电鳐"},
          {"label": "7", "value": "黄貂鱼"},
          {"label": "8", "value": "公鸡"},
          {"label": "9", "value": "母鸡"},
          {"label": "10", "value": "鸵鸟"},
          {"label": "11", "value": "燕雀"},
          {"label": "12", "value": "金翅雀"},
          {"label": "13", "value": "家朱雀"},
          {"label": "14", "value": "灯芯草雀"},
          {"label": "15", "value": "靛蓝雀,靛蓝鸟"},
          {"label": "16", "value": "蓝鹀"},
          {"label": "17", "value": "夜莺"},
          {"label": "18", "value": "松鸦"},
          {"label": "19", "value": "喜鹊"},
          {"label": "20", "value": "山雀"},
          {"label": "21", "value": "河鸟"},
          {"label": "22", "value": "鸢（猛禽）"},
          {"label": "23", "value": "秃头鹰"},
          {"label": "24", "value": "秃鹫"},
          {"label": "25", "value": "大灰猫头鹰"},
          {"label": "26", "value": "欧洲火蝾螈"},
          {"label": "27", "value": "普通蝾螈"},
          {"label": "28", "value": "水蜥"},
          {"label": "29", "value": "斑点蝾螈"},
          {"label": "30", "value": "蝾螈,泥狗"},
          {"label": "31", "value": "牛蛙"},
          {"label": "32", "value": "树蛙"},
          {"label": "33", "value": "尾蛙,铃蟾蜍,肋蟾蜍,尾蟾蜍"},
          {"label": "34", "value": "红海龟"},
          {"label": "35", "value": "皮革龟"},
          {"label": "36", "value": "泥龟"},
          {"label": "37", "value": "淡水龟"},
          {"label": "38", "value": "箱龟"},
          {"label": "39", "value": "带状壁虎"},
          {"label": "40", "value": "普通鬣蜥"},
          {"label": "41", "value": "美国变色龙"},
          {"label": "42", "value": "鞭尾蜥蜴"},
          {"label": "43", "value": "飞龙科蜥蜴"},
          {"label": "44", "value": "褶边蜥蜴"},
          {"label": "45", "value": "鳄鱼蜥蜴"},
          {"label": "46", "value": "毒蜥"},
          {"label": "47", "value": "绿蜥蜴"},
          {"label": "48", "value": "非洲变色龙"},
          {"label": "49", "value": "科莫多蜥蜴"},
          {"label": "50", "value": "非洲鳄,尼罗河鳄鱼"},
          {"label": "51", "value": "美国鳄鱼,鳄鱼"},
          {"label": "52", "value": "三角龙"},
          {"label": "53", "value": "雷蛇,蠕虫蛇"},
          {"label": "54", "value": "环蛇,环颈蛇"},
          {"label": "55", "value": "希腊蛇"},
          {"label": "56", "value": "绿蛇,草蛇"},
          {"label": "57", "value": "国王蛇"},
          {"label": "58", "value": "袜带蛇,草蛇"},
          {"label": "59", "value": "水蛇"},
          {"label": "60", "value": "藤蛇"},
          {"label": "61", "value": "夜蛇"},
          {"label": "62", "value": "大蟒蛇"},
          {"label": "63", "value": "岩石蟒蛇,岩蛇,蟒蛇"},
          {"label": "64", "value": "印度眼镜蛇"},
          {"label": "65", "value": "绿曼巴"},
          {"label": "66", "value": "海蛇"},
          {"label": "67", "value": "角腹蛇"},
          {"label": "68", "value": "菱纹响尾蛇"},
          {"label": "69", "value": "角响尾蛇"},
          {"label": "70", "value": "三叶虫"},
          {"label": "71", "value": "盲蜘蛛"},
          {"label": "72", "value": "蝎子"},
          {"label": "73", "value": "黑金花园蜘蛛"},
          {"label": "74", "value": "谷仓蜘蛛"},
          {"label": "75", "value": "花园蜘蛛"},
          {"label": "76", "value": "黑寡妇蜘蛛"},
          {"label": "77", "value": "狼蛛"},
          {"label": "78", "value": "狼蜘蛛,狩猎蜘蛛"},
          {"label": "79", "value": "壁虱"},
          {"label": "80", "value": "蜈蚣"},
          {"label": "81", "value": "黑松鸡"},
          {"label": "82", "value": "松鸡,雷鸟"},
          {"label": "83", "value": "披肩鸡,披肩榛鸡"},
          {"label": "84", "value": "草原鸡,草原松鸡"},
          {"label": "85", "value": "孔雀"},
          {"label": "86", "value": "鹌鹑"},
          {"label": "87", "value": "鹧鸪"},
          {"label": "88", "value": "非洲灰鹦鹉"},
          {"label": "89", "value": "金刚鹦鹉"},
          {"label": "90", "value": "硫冠鹦鹉"},
          {"label": "91", "value": "短尾鹦鹉"},
          {"label": "92", "value": "褐翅鸦鹃"},
          {"label": "93", "value": "蜜蜂"},
          {"label": "94", "value": "犀鸟"},
          {"label": "95", "value": "蜂鸟"},
          {"label": "96", "value": "鹟䴕"},
          {"label": "97", "value": "犀鸟"},
          {"label": "98", "value": "野鸭"},
          {"label": "99", "value": "红胸秋沙鸭"},
          {"label": "100", "value": "鹅"},
          {"label": "101", "value": "黑天鹅"},
          {"label": "102", "value": "大象"},
          {"label": "103", "value": "针鼹鼠"},
          {"label": "104", "value": "鸭嘴兽"},
          {"label": "105", "value": "沙袋鼠"},
          {"label": "106", "value": "考拉,考拉熊"},
          {"label": "107", "value": "袋熊"},
          {"label": "108", "value": "水母"},
          {"label": "109", "value": "海葵"},
          {"label": "110", "value": "脑珊瑚"},
          {"label": "111", "value": "扁形虫扁虫"},
          {"label": "112", "value": "线虫,蛔虫"},
          {"label": "113", "value": "海螺"},
          {"label": "114", "value": "蜗牛"},
          {"label": "115", "value": "鼻涕虫"},
          {"label": "116", "value": "海参"},
          {"label": "117", "value": "石鳖"},
          {"label": "118", "value": "鹦鹉螺"},
          {"label": "119", "value": "珍宝蟹"},
          {"label": "120", "value": "石蟹"},
          {"label": "121", "value": "招潮蟹"},
          {"label": "122", "value": "帝王蟹,阿拉斯加蟹,阿拉斯加帝王蟹"},
          {"label": "123", "value": "美国龙虾,缅因州龙虾"},
          {"label": "124", "value": "大螯虾"},
          {"label": "125", "value": "小龙虾"},
          {"label": "126", "value": "寄居蟹"},
          {"label": "127", "value": "等足目动物(明虾和螃蟹近亲)"},
          {"label": "128", "value": "白鹳"},
          {"label": "129", "value": "黑鹳"},
          {"label": "130", "value": "鹭"},
          {"label": "131", "value": "火烈鸟"},
          {"label": "132", "value": "小蓝鹭"},
          {"label": "133", "value": "美国鹭,大白鹭"},
          {"label": "134", "value": "麻鸦"},
          {"label": "135", "value": "鹤"},
          {"label": "136", "value": "秧鹤"},
          {"label": "137", "value": "欧洲水鸡,紫水鸡"},
          {"label": "138", "value": "沼泽泥母鸡,水母鸡"},
          {"label": "139", "value": "鸨"},
          {"label": "140", "value": "红翻石鹬"},
          {"label": "141", "value": "红背鹬,黑腹滨鹬"},
          {"label": "142", "value": "红脚鹬"},
          {"label": "143", "value": "半蹼鹬"},
          {"label": "144", "value": "蛎鹬"},
          {"label": "145", "value": "鹈鹕"},
          {"label": "146", "value": "国王企鹅"},
          {"label": "147", "value": "信天翁,大海鸟"},
          {"label": "148", "value": "灰鲸"},
          {"label": "149", "value": "杀人鲸,逆戟鲸,虎鲸"},
          {"label": "150", "value": "海牛"},
          {"label": "151", "value": "海狮"},
          {"label": "152", "value": "奇瓦瓦"},
          {"label": "153", "value": "日本猎犬"},
          {"label": "154", "value": "马尔济斯犬"},
          {"label": "155", "value": "狮子狗"},
          {"label": "156", "value": "西施犬"},
          {"label": "157", "value": "布莱尼姆猎犬"},
          {"label": "158", "value": "巴比狗"},
          {"label": "159", "value": "玩具犬"},
          {"label": "160", "value": "罗得西亚长背猎狗"},
          {"label": "161", "value": "阿富汗猎犬"},
          {"label": "162", "value": "猎犬"},
          {"label": "163", "value": "比格犬,猎兔犬"},
          {"label": "164", "value": "侦探犬"},
          {"label": "165", "value": "蓝色快狗"},
          {"label": "166", "value": "黑褐猎浣熊犬"},
          {"label": "167", "value": "沃克猎犬"},
          {"label": "168", "value": "英国猎狐犬"},
          {"label": "169", "value": "美洲赤狗"},
          {"label": "170", "value": "俄罗斯猎狼犬"},
          {"label": "171", "value": "爱尔兰猎狼犬"},
          {"label": "172", "value": "意大利灰狗"},
          {"label": "173", "value": "惠比特犬"},
          {"label": "174", "value": "依比沙猎犬"},
          {"label": "175", "value": "挪威猎犬"},
          {"label": "176", "value": "奥达猎犬,水獭猎犬"},
          {"label": "177", "value": "沙克犬,瞪羚猎犬"},
          {"label": "178", "value": "苏格兰猎鹿犬,猎鹿犬"},
          {"label": "179", "value": "威玛猎犬"},
          {"label": "180", "value": "斯塔福德郡牛头梗,斯塔福德郡斗牛梗"},
          {"label": "181", "value": "美国斯塔福德郡梗,美国比特斗牛梗,斗牛梗"},
          {"label": "182", "value": "贝德灵顿梗"},
          {"label": "183", "value": "边境梗"},
          {"label": "184", "value": "凯丽蓝梗"},
          {"label": "185", "value": "爱尔兰梗"},
          {"label": "186", "value": "诺福克梗"},
          {"label": "187", "value": "诺维奇梗"},
          {"label": "188", "value": "约克郡梗"},
          {"label": "189", "value": "刚毛猎狐梗"},
          {"label": "190", "value": "莱克兰梗"},
          {"label": "191", "value": "锡利哈姆梗"},
          {"label": "192", "value": "艾尔谷犬"},
          {"label": "193", "value": "凯恩梗"},
          {"label": "194", "value": "澳大利亚梗"},
          {"label": "195", "value": "丹迪丁蒙梗"},
          {"label": "196", "value": "波士顿梗"},
          {"label": "197", "value": "迷你雪纳瑞犬"},
          {"label": "198", "value": "巨型雪纳瑞犬"},
          {"label": "199", "value": "标准雪纳瑞犬"},
          {"label": "200", "value": "苏格兰梗"},
          {"label": "201", "value": "西藏梗,菊花狗"},
          {"label": "202", "value": "丝毛梗"},
          {"label": "203", "value": "软毛麦色梗"},
          {"label": "204", "value": "西高地白梗"},
          {"label": "205", "value": "拉萨阿普索犬"},
          {"label": "206", "value": "平毛寻回犬"},
          {"label": "207", "value": "卷毛寻回犬"},
          {"label": "208", "value": "金毛猎犬"},
          {"label": "209", "value": "拉布拉多猎犬"},
          {"label": "210", "value": "乞沙比克猎犬"},
          {"label": "211", "value": "德国短毛猎犬"},
          {"label": "212", "value": "维兹拉犬"},
          {"label": "213", "value": "英国谍犬"},
          {"label": "214", "value": "爱尔兰雪达犬,红色猎犬"},
          {"label": "215", "value": "戈登雪达犬"},
          {"label": "216", "value": "布列塔尼犬猎犬"},
          {"label": "217", "value": "黄毛,黄毛猎犬"},
          {"label": "218", "value": "英国史宾格犬"},
          {"label": "219", "value": "威尔士史宾格犬"},
          {"label": "220", "value": "可卡犬,英国可卡犬"},
          {"label": "221", "value": "萨塞克斯猎犬"},
          {"label": "222", "value": "爱尔兰水猎犬"},
          {"label": "223", "value": "哥威斯犬"},
          {"label": "224", "value": "舒柏奇犬"},
          {"label": "225", "value": "比利时牧羊犬"},
          {"label": "226", "value": "马里努阿犬"},
          {"label": "227", "value": "伯瑞犬"},
          {"label": "228", "value": "凯尔皮犬"},
          {"label": "229", "value": "匈牙利牧羊犬"},
          {"label": "230", "value": "老英国牧羊犬"},
          {"label": "231", "value": "喜乐蒂牧羊犬"},
          {"label": "232", "value": "牧羊犬"},
          {"label": "233", "value": "边境牧羊犬"},
          {"label": "234", "value": "法兰德斯牧牛狗"},
          {"label": "235", "value": "罗特韦尔犬"},
          {"label": "236", "value": "德国牧羊犬,德国警犬,阿尔萨斯"},
          {"label": "237", "value": "多伯曼犬,杜宾犬"},
          {"label": "238", "value": "迷你杜宾犬"},
          {"label": "239", "value": "大瑞士山地犬"},
          {"label": "240", "value": "伯恩山犬"},
          {"label": "241", "value": "Appenzeller狗"},
          {"label": "242", "value": "EntleBucher狗"},
          {"label": "243", "value": "拳师狗"},
          {"label": "244", "value": "斗牛獒"},
          {"label": "245", "value": "藏獒"},
          {"label": "246", "value": "法国斗牛犬"},
          {"label": "247", "value": "大丹犬"},
          {"label": "248", "value": "圣伯纳德狗"},
          {"label": "249", "value": "爱斯基摩犬,哈士奇"},
          {"label": "250", "value": "雪橇犬,阿拉斯加爱斯基摩狗"},
          {"label": "251", "value": "哈士奇"},
          {"label": "252", "value": "达尔马提亚,教练车狗"},
          {"label": "253", "value": "狮毛狗"},
          {"label": "254", "value": "巴辛吉狗"},
          {"label": "255", "value": "哈巴狗,狮子狗"},
          {"label": "256", "value": "莱昂贝格狗"},
          {"label": "257", "value": "纽芬兰岛狗"},
          {"label": "258", "value": "大白熊犬"},
          {"label": "259", "value": "萨摩耶犬"},
          {"label": "260", "value": "博美犬"},
          {"label": "261", "value": "松狮,松狮"},
          {"label": "262", "value": "荷兰卷尾狮毛狗"},
          {"label": "263", "value": "布鲁塞尔格林芬犬"},
          {"label": "264", "value": "彭布洛克威尔士科基犬"},
          {"label": "265", "value": "威尔士柯基犬"},
          {"label": "266", "value": "玩具贵宾犬"},
          {"label": "267", "value": "迷你贵宾犬"},
          {"label": "268", "value": "标准贵宾犬"},
          {"label": "269", "value": "墨西哥无毛犬"},
          {"label": "270", "value": "灰狼"},
          {"label": "271", "value": "白狼,北极狼"},
          {"label": "272", "value": "红太狼,鬃狼,犬犬鲁弗斯"},
          {"label": "273", "value": "狼,草原狼,刷狼,郊狼"},
          {"label": "274", "value": "澳洲野狗,澳大利亚野犬"},
          {"label": "275", "value": "豺"},
          {"label": "276", "value": "非洲猎犬,土狼犬"},
          {"label": "277", "value": "鬣狗"},
          {"label": "278", "value": "红狐狸"},
          {"label": "279", "value": "沙狐"},
          {"label": "280", "value": "北极狐狸,白狐狸"},
          {"label": "281", "value": "灰狐狸"},
          {"label": "282", "value": "虎斑猫"},
          {"label": "283", "value": "山猫,虎猫"},
          {"label": "284", "value": "波斯猫"},
          {"label": "285", "value": "暹罗暹罗猫,"},
          {"label": "286", "value": "埃及猫"},
          {"label": "287", "value": "美洲狮,美洲豹"},
          {"label": "288", "value": "猞猁,山猫"},
          {"label": "289", "value": "豹子"},
          {"label": "290", "value": "雪豹"},
          {"label": "291", "value": "美洲虎"},
          {"label": "292", "value": "狮子"},
          {"label": "293", "value": "老虎"},
          {"label": "294", "value": "猎豹"},
          {"label": "295", "value": "棕熊"},
          {"label": "296", "value": "美洲黑熊"},
          {"label": "297", "value": "冰熊,北极熊"},
          {"label": "298", "value": "懒熊"},
          {"label": "299", "value": "猫鼬"},
          {"label": "300", "value": "猫鼬,海猫"},
          {"label": "301", "value": "虎甲虫"},
          {"label": "302", "value": "瓢虫"},
          {"label": "303", "value": "土鳖虫"},
          {"label": "304", "value": "天牛"},
          {"label": "305", "value": "龟甲虫"},
          {"label": "306", "value": "粪甲虫"},
          {"label": "307", "value": "犀牛甲虫"},
          {"label": "308", "value": "象甲"},
          {"label": "309", "value": "苍蝇"},
          {"label": "310", "value": "蜜蜂"},
          {"label": "311", "value": "蚂蚁"},
          {"label": "312", "value": "蚱蜢"},
          {"label": "313", "value": "蟋蟀"},
          {"label": "314", "value": "竹节虫"},
          {"label": "315", "value": "蟑螂"},
          {"label": "316", "value": "螳螂"},
          {"label": "317", "value": "蝉"},
          {"label": "318", "value": "叶蝉"},
          {"label": "319", "value": "草蜻蛉"},
          {"label": "320", "value": "蜻蜓"},
          {"label": "321", "value": "豆娘,蜻蛉"},
          {"label": "322", "value": "优红蛱蝶"},
          {"label": "323", "value": "小环蝴蝶"},
          {"label": "324", "value": "君主蝴蝶,大斑蝶"},
          {"label": "325", "value": "菜粉蝶"},
          {"label": "326", "value": "白蝴蝶"},
          {"label": "327", "value": "灰蝶"},
          {"label": "328", "value": "海星"},
          {"label": "329", "value": "海胆"},
          {"label": "330", "value": "海参,海黄瓜"},
          {"label": "331", "value": "野兔"},
          {"label": "332", "value": "兔"},
          {"label": "333", "value": "安哥拉兔"},
          {"label": "334", "value": "仓鼠"},
          {"label": "335", "value": "刺猬,豪猪,"},
          {"label": "336", "value": "黑松鼠"},
          {"label": "337", "value": "土拨鼠"},
          {"label": "338", "value": "海狸"},
          {"label": "339", "value": "豚鼠,豚鼠"},
          {"label": "340", "value": "栗色马"},
          {"label": "341", "value": "斑马"},
          {"label": "342", "value": "猪"},
          {"label": "343", "value": "野猪"},
          {"label": "344", "value": "疣猪"},
          {"label": "345", "value": "河马"},
          {"label": "346", "value": "牛"},
          {"label": "347", "value": "水牛,亚洲水牛"},
          {"label": "348", "value": "野牛"},
          {"label": "349", "value": "公羊"},
          {"label": "350", "value": "大角羊,洛矶山大角羊"},
          {"label": "351", "value": "山羊"},
          {"label": "352", "value": "狷羚"},
          {"label": "353", "value": "黑斑羚"},
          {"label": "354", "value": "瞪羚"},
          {"label": "355", "value": "阿拉伯单峰骆驼,骆驼"},
          {"label": "356", "value": "骆驼"},
          {"label": "357", "value": "黄鼠狼"},
          {"label": "358", "value": "水貂"},
          {"label": "359", "value": "臭猫"},
          {"label": "360", "value": "黑足鼬"},
          {"label": "361", "value": "水獭"},
          {"label": "362", "value": "臭鼬,木猫"},
          {"label": "363", "value": "獾"},
          {"label": "364", "value": "犰狳"},
          {"label": "365", "value": "树懒"},
          {"label": "366", "value": "猩猩,婆罗洲猩猩"},
          {"label": "367", "value": "大猩猩"},
          {"label": "368", "value": "黑猩猩"},
          {"label": "369", "value": "长臂猿"},
          {"label": "370", "value": "合趾猿长臂猿,合趾猿"},
          {"label": "371", "value": "长尾猴"},
          {"label": "372", "value": "赤猴"},
          {"label": "373", "value": "狒狒"},
          {"label": "374", "value": "恒河猴,猕猴"},
          {"label": "375", "value": "白头叶猴"},
          {"label": "376", "value": "疣猴"},
          {"label": "377", "value": "长鼻猴"},
          {"label": "378", "value": "狨（美洲产小型长尾猴）"},
          {"label": "379", "value": "卷尾猴"},
          {"label": "380", "value": "吼猴"},
          {"label": "381", "value": "伶猴"},
          {"label": "382", "value": "蜘蛛猴"},
          {"label": "383", "value": "松鼠猴"},
          {"label": "384", "value": "马达加斯加环尾狐猴,鼠狐猴"},
          {"label": "385", "value": "大狐猴,马达加斯加大狐猴"},
          {"label": "386", "value": "印度大象,亚洲象"},
          {"label": "387", "value": "非洲象,非洲象"},
          {"label": "388", "value": "小熊猫"},
          {"label": "389", "value": "大熊猫"},
          {"label": "390", "value": "杖鱼"},
          {"label": "391", "value": "鳗鱼"},
          {"label": "392", "value": "银鲑,银鲑鱼"},
          {"label": "393", "value": "三色刺蝶鱼"},
          {"label": "394", "value": "海葵鱼"},
          {"label": "395", "value": "鲟鱼"},
          {"label": "396", "value": "雀鳝"},
          {"label": "397", "value": "狮子鱼"},
          {"label": "398", "value": "河豚"},
          {"label": "399", "value": "算盘"},
          {"label": "400", "value": "长袍"},
          {"label": "401", "value": "学位袍"},
          {"label": "402", "value": "手风琴"},
          {"label": "403", "value": "原声吉他"},
          {"label": "404", "value": "航空母舰"},
          {"label": "405", "value": "客机"},
          {"label": "406", "value": "飞艇"},
          {"label": "407", "value": "祭坛"},
          {"label": "408", "value": "救护车"},
          {"label": "409", "value": "水陆两用车"},
          {"label": "410", "value": "模拟时钟"},
          {"label": "411", "value": "蜂房"},
          {"label": "412", "value": "围裙"},
          {"label": "413", "value": "垃圾桶"},
          {"label": "414", "value": "攻击步枪,枪"},
          {"label": "415", "value": "背包"},
          {"label": "416", "value": "面包店,面包铺,"},
          {"label": "417", "value": "平衡木"},
          {"label": "418", "value": "热气球"},
          {"label": "419", "value": "圆珠笔"},
          {"label": "420", "value": "创可贴"},
          {"label": "421", "value": "班卓琴"},
          {"label": "422", "value": "栏杆,楼梯扶手"},
          {"label": "423", "value": "杠铃"},
          {"label": "424", "value": "理发师的椅子"},
          {"label": "425", "value": "理发店"},
          {"label": "426", "value": "牲口棚"},
          {"label": "427", "value": "晴雨表"},
          {"label": "428", "value": "圆筒"},
          {"label": "429", "value": "园地小车,手推车"},
          {"label": "430", "value": "棒球"},
          {"label": "431", "value": "篮球"},
          {"label": "432", "value": "婴儿床"},
          {"label": "433", "value": "巴松管,低音管"},
          {"label": "434", "value": "游泳帽"},
          {"label": "435", "value": "沐浴毛巾"},
          {"label": "436", "value": "浴缸,澡盆"},
          {"label": "437", "value": "沙滩车,旅行车"},
          {"label": "438", "value": "灯塔"},
          {"label": "439", "value": "高脚杯"},
          {"label": "440", "value": "熊皮高帽"},
          {"label": "441", "value": "啤酒瓶"},
          {"label": "442", "value": "啤酒杯 "},
          {"label": "443", "value": "钟塔"},
          {"label": "444", "value": "（小儿用的）围嘴"},
          {"label": "445", "value": "串联自行车,"},
          {"label": "446", "value": "比基尼"},
          {"label": "447", "value": "装订册"},
          {"label": "448", "value": "双筒望远镜"},
          {"label": "449", "value": "鸟舍"},
          {"label": "450", "value": "船库"},
          {"label": "451", "value": "雪橇"},
          {"label": "452", "value": "饰扣式领带"},
          {"label": "453", "value": "阔边女帽"},
          {"label": "454", "value": "书橱"},
          {"label": "455", "value": "书店,书摊"},
          {"label": "456", "value": "瓶盖"},
          {"label": "457", "value": "弓箭"},
          {"label": "458", "value": "蝴蝶结领结"},
          {"label": "459", "value": "铜制牌位"},
          {"label": "460", "value": "奶罩"},
          {"label": "461", "value": "防波堤,海堤"},
          {"label": "462", "value": "铠甲"},
          {"label": "463", "value": "扫帚"},
          {"label": "464", "value": "桶"},
          {"label": "465", "value": "扣环"},
          {"label": "466", "value": "防弹背心"},
          {"label": "467", "value": "动车,子弹头列车"},
          {"label": "468", "value": "肉铺,肉菜市场"},
          {"label": "469", "value": "出租车"},
          {"label": "470", "value": "大锅"},
          {"label": "471", "value": "蜡烛"},
          {"label": "472", "value": "大炮"},
          {"label": "473", "value": "独木舟"},
          {"label": "474", "value": "开瓶器,开罐器"},
          {"label": "475", "value": "开衫"},
          {"label": "476", "value": "车镜"},
          {"label": "477", "value": "旋转木马"},
          {"label": "478", "value": "木匠的工具包,工具包"},
          {"label": "479", "value": "纸箱"},
          {"label": "480", "value": "车轮"},
          {"label": "481", "value": "取款机,自动取款机"},
          {"label": "482", "value": "盒式录音带"},
          {"label": "483", "value": "卡带播放器"},
          {"label": "484", "value": "城堡"},
          {"label": "485", "value": "双体船"},
          {"label": "486", "value": "CD播放器"},
          {"label": "487", "value": "大提琴"},
          {"label": "488", "value": "移动电话,手机"},
          {"label": "489", "value": "铁链"},
          {"label": "490", "value": "围栏"},
          {"label": "491", "value": "链甲"},
          {"label": "492", "value": "电锯,油锯"},
          {"label": "493", "value": "箱子"},
          {"label": "494", "value": "衣柜,洗脸台"},
          {"label": "495", "value": "编钟,钟,锣"},
          {"label": "496", "value": "中国橱柜"},
          {"label": "497", "value": "圣诞袜"},
          {"label": "498", "value": "教堂,教堂建筑"},
          {"label": "499", "value": "电影院,剧场"},
          {"label": "500", "value": "切肉刀,菜刀"},
          {"label": "501", "value": "悬崖屋"},
          {"label": "502", "value": "斗篷"},
          {"label": "503", "value": "木屐,木鞋"},
          {"label": "504", "value": "鸡尾酒调酒器"},
          {"label": "505", "value": "咖啡杯"},
          {"label": "506", "value": "咖啡壶"},
          {"label": "507", "value": "螺旋结构（楼梯）"},
          {"label": "508", "value": "组合锁"},
          {"label": "509", "value": "电脑键盘,键盘"},
          {"label": "510", "value": "糖果,糖果店"},
          {"label": "511", "value": "集装箱船"},
          {"label": "512", "value": "敞篷车"},
          {"label": "513", "value": "开瓶器,瓶螺杆"},
          {"label": "514", "value": "短号,喇叭"},
          {"label": "515", "value": "牛仔靴"},
          {"label": "516", "value": "牛仔帽"},
          {"label": "517", "value": "摇篮"},
          {"label": "518", "value": "起重机"},
          {"label": "519", "value": "头盔"},
          {"label": "520", "value": "板条箱"},
          {"label": "521", "value": "小儿床"},
          {"label": "522", "value": "砂锅"},
          {"label": "523", "value": "槌球"},
          {"label": "524", "value": "拐杖"},
          {"label": "525", "value": "胸甲"},
          {"label": "526", "value": "大坝,堤防"},
          {"label": "527", "value": "书桌"},
          {"label": "528", "value": "台式电脑"},
          {"label": "529", "value": "有线电话"},
          {"label": "530", "value": "尿布湿"},
          {"label": "531", "value": "数字时钟"},
          {"label": "532", "value": "数字手表"},
          {"label": "533", "value": "餐桌板"},
          {"label": "534", "value": "抹布"},
          {"label": "535", "value": "洗碗机,洗碟机"},
          {"label": "536", "value": "盘式制动器"},
          {"label": "537", "value": "码头,船坞,码头设施"},
          {"label": "538", "value": "狗拉雪橇"},
          {"label": "539", "value": "圆顶"},
          {"label": "540", "value": "门垫,垫子"},
          {"label": "541", "value": "钻井平台,海上钻井"},
          {"label": "542", "value": "鼓,乐器,鼓膜"},
          {"label": "543", "value": "鼓槌"},
          {"label": "544", "value": "哑铃"},
          {"label": "545", "value": "荷兰烤箱"},
          {"label": "546", "value": "电风扇,鼓风机"},
          {"label": "547", "value": "电吉他"},
          {"label": "548", "value": "电力机车"},
          {"label": "549", "value": "电视,电视柜"},
          {"label": "550", "value": "信封"},
          {"label": "551", "value": "浓缩咖啡机"},
          {"label": "552", "value": "扑面粉"},
          {"label": "553", "value": "女用长围巾"},
          {"label": "554", "value": "文件,文件柜,档案柜"},
          {"label": "555", "value": "消防船"},
          {"label": "556", "value": "消防车"},
          {"label": "557", "value": "火炉栏"},
          {"label": "558", "value": "旗杆"},
          {"label": "559", "value": "长笛"},
          {"label": "560", "value": "折叠椅"},
          {"label": "561", "value": "橄榄球头盔"},
          {"label": "562", "value": "叉车"},
          {"label": "563", "value": "喷泉"},
          {"label": "564", "value": "钢笔"},
          {"label": "565", "value": "有四根帷柱的床"},
          {"label": "566", "value": "运货车厢"},
          {"label": "567", "value": "圆号,喇叭"},
          {"label": "568", "value": "煎锅"},
          {"label": "569", "value": "裘皮大衣"},
          {"label": "570", "value": "垃圾车"},
          {"label": "571", "value": "防毒面具,呼吸器"},
          {"label": "572", "value": "汽油泵"},
          {"label": "573", "value": "高脚杯"},
          {"label": "574", "value": "卡丁车"},
          {"label": "575", "value": "高尔夫球"},
          {"label": "576", "value": "高尔夫球车"},
          {"label": "577", "value": "狭长小船"},
          {"label": "578", "value": "锣"},
          {"label": "579", "value": "礼服"},
          {"label": "580", "value": "钢琴"},
          {"label": "581", "value": "温室,苗圃"},
          {"label": "582", "value": "散热器格栅"},
          {"label": "583", "value": "杂货店,食品市场"},
          {"label": "584", "value": "断头台"},
          {"label": "585", "value": "小发夹"},
          {"label": "586", "value": "头发喷雾"},
          {"label": "587", "value": "半履带装甲车"},
          {"label": "588", "value": "锤子"},
          {"label": "589", "value": "大篮子"},
          {"label": "590", "value": "手摇鼓风机,吹风机"},
          {"label": "591", "value": "手提电脑"},
          {"label": "592", "value": "手帕"},
          {"label": "593", "value": "硬盘"},
          {"label": "594", "value": "口琴,口风琴"},
          {"label": "595", "value": "竖琴"},
          {"label": "596", "value": "收割机"},
          {"label": "597", "value": "斧头"},
          {"label": "598", "value": "手枪皮套"},
          {"label": "599", "value": "家庭影院"},
          {"label": "600", "value": "蜂窝"},
          {"label": "601", "value": "钩爪"},
          {"label": "602", "value": "衬裙"},
          {"label": "603", "value": "单杠"},
          {"label": "604", "value": "马车"},
          {"label": "605", "value": "沙漏"},
          {"label": "606", "value": "iPod"},
          {"label": "607", "value": "熨斗"},
          {"label": "608", "value": "南瓜灯笼"},
          {"label": "609", "value": "牛仔裤,蓝色牛仔裤"},
          {"label": "610", "value": "吉普车"},
          {"label": "611", "value": "运动衫,T恤"},
          {"label": "612", "value": "拼图"},
          {"label": "613", "value": "人力车"},
          {"label": "614", "value": "操纵杆"},
          {"label": "615", "value": "和服"},
          {"label": "616", "value": "护膝"},
          {"label": "617", "value": "蝴蝶结"},
          {"label": "618", "value": "大褂,实验室外套"},
          {"label": "619", "value": "长柄勺"},
          {"label": "620", "value": "灯罩"},
          {"label": "621", "value": "笔记本电脑"},
          {"label": "622", "value": "割草机"},
          {"label": "623", "value": "镜头盖"},
          {"label": "624", "value": "开信刀,裁纸刀"},
          {"label": "625", "value": "图书馆"},
          {"label": "626", "value": "救生艇"},
          {"label": "627", "value": "点火器,打火机"},
          {"label": "628", "value": "豪华轿车"},
          {"label": "629", "value": "远洋班轮"},
          {"label": "630", "value": "唇膏,口红"},
          {"label": "631", "value": "平底便鞋"},
          {"label": "632", "value": "洗剂"},
          {"label": "633", "value": "扬声器"},
          {"label": "634", "value": "放大镜"},
          {"label": "635", "value": "锯木厂"},
          {"label": "636", "value": "磁罗盘"},
          {"label": "637", "value": "邮袋"},
          {"label": "638", "value": "信箱"},
          {"label": "639", "value": "女游泳衣"},
          {"label": "640", "value": "有肩带浴衣"},
          {"label": "641", "value": "窨井盖"},
          {"label": "642", "value": "沙球（一种打击乐器）"},
          {"label": "643", "value": "马林巴木琴"},
          {"label": "644", "value": "面膜"},
          {"label": "645", "value": "火柴"},
          {"label": "646", "value": "花柱"},
          {"label": "647", "value": "迷宫"},
          {"label": "648", "value": "量杯"},
          {"label": "649", "value": "药箱"},
          {"label": "650", "value": "巨石,巨石结构"},
          {"label": "651", "value": "麦克风"},
          {"label": "652", "value": "微波炉"},
          {"label": "653", "value": "军装"},
          {"label": "654", "value": "奶桶"},
          {"label": "655", "value": "迷你巴士"},
          {"label": "656", "value": "迷你裙"},
          {"label": "657", "value": "面包车"},
          {"label": "658", "value": "导弹"},
          {"label": "659", "value": "连指手套"},
          {"label": "660", "value": "搅拌钵"},
          {"label": "661", "value": "活动房屋（由汽车拖拉的）"},
          {"label": "662", "value": "T型发动机小汽车"},
          {"label": "663", "value": "调制解调器"},
          {"label": "664", "value": "修道院"},
          {"label": "665", "value": "显示器"},
          {"label": "666", "value": "电瓶车"},
          {"label": "667", "value": "砂浆"},
          {"label": "668", "value": "学士"},
          {"label": "669", "value": "清真寺"},
          {"label": "670", "value": "蚊帐"},
          {"label": "671", "value": "摩托车"},
          {"label": "672", "value": "山地自行车"},
          {"label": "673", "value": "登山帐"},
          {"label": "674", "value": "鼠标,电脑鼠标"},
          {"label": "675", "value": "捕鼠器"},
          {"label": "676", "value": "搬家车"},
          {"label": "677", "value": "口套"},
          {"label": "678", "value": "钉子"},
          {"label": "679", "value": "颈托"},
          {"label": "680", "value": "项链"},
          {"label": "681", "value": "乳头（瓶）"},
          {"label": "682", "value": "笔记本,笔记本电脑"},
          {"label": "683", "value": "方尖碑"},
          {"label": "684", "value": "双簧管"},
          {"label": "685", "value": "陶笛,卵形笛"},
          {"label": "686", "value": "里程表"},
          {"label": "687", "value": "滤油器"},
          {"label": "688", "value": "风琴,管风琴"},
          {"label": "689", "value": "示波器"},
          {"label": "690", "value": "罩裙"},
          {"label": "691", "value": "牛车"},
          {"label": "692", "value": "氧气面罩"},
          {"label": "693", "value": "包装"},
          {"label": "694", "value": "船桨"},
          {"label": "695", "value": "明轮,桨轮"},
          {"label": "696", "value": "挂锁,扣锁"},
          {"label": "697", "value": "画笔"},
          {"label": "698", "value": "睡衣"},
          {"label": "699", "value": "宫殿"},
          {"label": "700", "value": "排箫,鸣管"},
          {"label": "701", "value": "纸巾"},
          {"label": "702", "value": "降落伞"},
          {"label": "703", "value": "双杠"},
          {"label": "704", "value": "公园长椅"},
          {"label": "705", "value": "停车收费表,停车计时器"},
          {"label": "706", "value": "客车,教练车"},
          {"label": "707", "value": "露台,阳台"},
          {"label": "708", "value": "付费电话"},
          {"label": "709", "value": "基座,基脚"},
          {"label": "710", "value": "铅笔盒"},
          {"label": "711", "value": "卷笔刀"},
          {"label": "712", "value": "香水（瓶）"},
          {"label": "713", "value": "培养皿"},
          {"label": "714", "value": "复印机"},
          {"label": "715", "value": "拨弦片,拨子"},
          {"label": "716", "value": "尖顶头盔"},
          {"label": "717", "value": "栅栏,栅栏"},
          {"label": "718", "value": "皮卡,皮卡车"},
          {"label": "719", "value": "桥墩"},
          {"label": "720", "value": "存钱罐"},
          {"label": "721", "value": "药瓶"},
          {"label": "722", "value": "枕头"},
          {"label": "723", "value": "乒乓球"},
          {"label": "724", "value": "风车"},
          {"label": "725", "value": "海盗船"},
          {"label": "726", "value": "水罐"},
          {"label": "727", "value": "木工刨"},
          {"label": "728", "value": "天文馆"},
          {"label": "729", "value": "塑料袋"},
          {"label": "730", "value": "板架"},
          {"label": "731", "value": "犁型铲雪机"},
          {"label": "732", "value": "手压皮碗泵"},
          {"label": "733", "value": "宝丽来相机"},
          {"label": "734", "value": "电线杆"},
          {"label": "735", "value": "警车,巡逻车"},
          {"label": "736", "value": "雨披"},
          {"label": "737", "value": "台球桌"},
          {"label": "738", "value": "充气饮料瓶"},
          {"label": "739", "value": "花盆"},
          {"label": "740", "value": "陶工旋盘"},
          {"label": "741", "value": "电钻"},
          {"label": "742", "value": "祈祷垫,地毯"},
          {"label": "743", "value": "打印机"},
          {"label": "744", "value": "监狱"},
          {"label": "745", "value": "炮弹,导弹"},
          {"label": "746", "value": "投影仪"},
          {"label": "747", "value": "冰球"},
          {"label": "748", "value": "沙包,吊球"},
          {"label": "749", "value": "钱包"},
          {"label": "750", "value": "羽管笔"},
          {"label": "751", "value": "被子"},
          {"label": "752", "value": "赛车"},
          {"label": "753", "value": "球拍"},
          {"label": "754", "value": "散热器"},
          {"label": "755", "value": "收音机"},
          {"label": "756", "value": "射电望远镜,无线电反射器"},
          {"label": "757", "value": "雨桶"},
          {"label": "758", "value": "休闲车,房车"},
          {"label": "759", "value": "卷轴,卷筒"},
          {"label": "760", "value": "反射式照相机"},
          {"label": "761", "value": "冰箱,冰柜"},
          {"label": "762", "value": "遥控器"},
          {"label": "763", "value": "餐厅,饮食店,食堂"},
          {"label": "764", "value": "左轮手枪"},
          {"label": "765", "value": "步枪"},
          {"label": "766", "value": "摇椅"},
          {"label": "767", "value": "电转烤肉架"},
          {"label": "768", "value": "橡皮"},
          {"label": "769", "value": "橄榄球"},
          {"label": "770", "value": "直尺"},
          {"label": "771", "value": "跑步鞋"},
          {"label": "772", "value": "保险柜"},
          {"label": "773", "value": "安全别针"},
          {"label": "774", "value": "盐瓶（调味用）"},
          {"label": "775", "value": "凉鞋"},
          {"label": "776", "value": "纱笼,围裙"},
          {"label": "777", "value": "萨克斯管"},
          {"label": "778", "value": "剑鞘"},
          {"label": "779", "value": "秤,称重机"},
          {"label": "780", "value": "校车"},
          {"label": "781", "value": "帆船"},
          {"label": "782", "value": "记分牌"},
          {"label": "783", "value": "屏幕"},
          {"label": "784", "value": "螺丝"},
          {"label": "785", "value": "螺丝刀"},
          {"label": "786", "value": "安全带"},
          {"label": "787", "value": "缝纫机"},
          {"label": "788", "value": "盾牌,盾牌"},
          {"label": "789", "value": "皮鞋店,鞋店"},
          {"label": "790", "value": "障子"},
          {"label": "791", "value": "购物篮"},
          {"label": "792", "value": "购物车"},
          {"label": "793", "value": "铁锹"},
          {"label": "794", "value": "浴帽"},
          {"label": "795", "value": "浴帘"},
          {"label": "796", "value": "滑雪板"},
          {"label": "797", "value": "滑雪面罩"},
          {"label": "798", "value": "睡袋"},
          {"label": "799", "value": "滑尺"},
          {"label": "800", "value": "滑动门"},
          {"label": "801", "value": "角子老虎机"},
          {"label": "802", "value": "潜水通气管"},
          {"label": "803", "value": "雪橇"},
          {"label": "804", "value": "扫雪机,扫雪机"},
          {"label": "805", "value": "皂液器"},
          {"label": "806", "value": "足球"},
          {"label": "807", "value": "袜子"},
          {"label": "808", "value": "碟式太阳能,太阳能集热器,太阳能炉"},
          {"label": "809", "value": "宽边帽"},
          {"label": "810", "value": "汤碗"},
          {"label": "811", "value": "空格键"},
          {"label": "812", "value": "空间加热器"},
          {"label": "813", "value": "航天飞机"},
          {"label": "814", "value": "铲（搅拌或涂敷用的）"},
          {"label": "815", "value": "快艇"},
          {"label": "816", "value": "蜘蛛网"},
          {"label": "817", "value": "纺锤,纱锭"},
          {"label": "818", "value": "跑车"},
          {"label": "819", "value": "聚光灯"},
          {"label": "820", "value": "舞台"},
          {"label": "821", "value": "蒸汽机车"},
          {"label": "822", "value": "钢拱桥"},
          {"label": "823", "value": "钢滚筒"},
          {"label": "824", "value": "听诊器"},
          {"label": "825", "value": "女用披肩"},
          {"label": "826", "value": "石头墙"},
          {"label": "827", "value": "秒表"},
          {"label": "828", "value": "火炉"},
          {"label": "829", "value": "过滤器"},
          {"label": "830", "value": "有轨电车,电车"},
          {"label": "831", "value": "担架"},
          {"label": "832", "value": "沙发床"},
          {"label": "833", "value": "佛塔"},
          {"label": "834", "value": "潜艇,潜水艇"},
          {"label": "835", "value": "套装,衣服"},
          {"label": "836", "value": "日晷"},
          {"label": "837", "value": "太阳镜"},
          {"label": "838", "value": "太阳镜,墨镜"},
          {"label": "839", "value": "防晒霜,防晒剂"},
          {"label": "840", "value": "悬索桥"},
          {"label": "841", "value": "拖把"},
          {"label": "842", "value": "运动衫"},
          {"label": "843", "value": "游泳裤"},
          {"label": "844", "value": "秋千"},
          {"label": "845", "value": "开关,电器开关"},
          {"label": "846", "value": "注射器"},
          {"label": "847", "value": "台灯"},
          {"label": "848", "value": "坦克,装甲战车,装甲战斗车辆"},
          {"label": "849", "value": "磁带播放器"},
          {"label": "850", "value": "茶壶"},
          {"label": "851", "value": "泰迪,泰迪熊"},
          {"label": "852", "value": "电视"},
          {"label": "853", "value": "网球"},
          {"label": "854", "value": "茅草,茅草屋顶"},
          {"label": "855", "value": "幕布,剧院的帷幕"},
          {"label": "856", "value": "顶针"},
          {"label": "857", "value": "脱粒机"},
          {"label": "858", "value": "宝座"},
          {"label": "859", "value": "瓦屋顶"},
          {"label": "860", "value": "烤面包机"},
          {"label": "861", "value": "烟草店,烟草"},
          {"label": "862", "value": "马桶"},
          {"label": "863", "value": "火炬"},
          {"label": "864", "value": "图腾柱"},
          {"label": "865", "value": "拖车,牵引车,清障车"},
          {"label": "866", "value": "玩具店"},
          {"label": "867", "value": "拖拉机"},
          {"label": "868", "value": "拖车,铰接式卡车"},
          {"label": "869", "value": "托盘"},
          {"label": "870", "value": "风衣"},
          {"label": "871", "value": "三轮车"},
          {"label": "872", "value": "三体船"},
          {"label": "873", "value": "三脚架"},
          {"label": "874", "value": "凯旋门"},
          {"label": "875", "value": "无轨电车"},
          {"label": "876", "value": "长号"},
          {"label": "877", "value": "浴盆,浴缸"},
          {"label": "878", "value": "旋转式栅门"},
          {"label": "879", "value": "打字机键盘"},
          {"label": "880", "value": "伞"},
          {"label": "881", "value": "独轮车"},
          {"label": "882", "value": "直立式钢琴"},
          {"label": "883", "value": "真空吸尘器"},
          {"label": "884", "value": "花瓶"},
          {"label": "885", "value": "拱顶"},
          {"label": "886", "value": "天鹅绒"},
          {"label": "887", "value": "自动售货机"},
          {"label": "888", "value": "祭服"},
          {"label": "889", "value": "高架桥"},
          {"label": "890", "value": "小提琴,小提琴"},
          {"label": "891", "value": "排球"},
          {"label": "892", "value": "松饼机"},
          {"label": "893", "value": "挂钟"},
          {"label": "894", "value": "钱包,皮夹"},
          {"label": "895", "value": "衣柜,壁橱"},
          {"label": "896", "value": "军用飞机"},
          {"label": "897", "value": "洗脸盆,洗手盆"},
          {"label": "898", "value": "洗衣机,自动洗衣机"},
          {"label": "899", "value": "水瓶"},
          {"label": "900", "value": "水壶"},
          {"label": "901", "value": "水塔"},
          {"label": "902", "value": "威士忌壶"},
          {"label": "903", "value": "哨子"},
          {"label": "904", "value": "假发"},
          {"label": "905", "value": "纱窗"},
          {"label": "906", "value": "百叶窗"},
          {"label": "907", "value": "温莎领带"},
          {"label": "908", "value": "葡萄酒瓶"},
          {"label": "909", "value": "飞机翅膀,飞机"},
          {"label": "910", "value": "炒菜锅"},
          {"label": "911", "value": "木制的勺子"},
          {"label": "912", "value": "毛织品,羊绒"},
          {"label": "913", "value": "栅栏,围栏"},
          {"label": "914", "value": "沉船"},
          {"label": "915", "value": "双桅船"},
          {"label": "916", "value": "蒙古包"},
          {"label": "917", "value": "网站,互联网网站"},
          {"label": "918", "value": "漫画"},
          {"label": "919", "value": "纵横字谜"},
          {"label": "920", "value": "路标"},
          {"label": "921", "value": "交通信号灯"},
          {"label": "922", "value": "防尘罩,书皮"},
          {"label": "923", "value": "菜单"},
          {"label": "924", "value": "盘子"},
          {"label": "925", "value": "鳄梨酱"},
          {"label": "926", "value": "清汤"},
          {"label": "927", "value": "罐焖土豆烧肉"},
          {"label": "928", "value": "蛋糕"},
          {"label": "929", "value": "冰淇淋"},
          {"label": "930", "value": "雪糕,冰棍,冰棒"},
          {"label": "931", "value": "法式面包"},
          {"label": "932", "value": "百吉饼"},
          {"label": "933", "value": "椒盐脆饼"},
          {"label": "934", "value": "芝士汉堡"},
          {"label": "935", "value": "热狗"},
          {"label": "936", "value": "土豆泥"},
          {"label": "937", "value": "结球甘蓝"},
          {"label": "938", "value": "西兰花"},
          {"label": "939", "value": "菜花"},
          {"label": "940", "value": "绿皮密生西葫芦"},
          {"label": "941", "value": "西葫芦"},
          {"label": "942", "value": "小青南瓜"},
          {"label": "943", "value": "南瓜"},
          {"label": "944", "value": "黄瓜"},
          {"label": "945", "value": "朝鲜蓟"},
          {"label": "946", "value": "甜椒"},
          {"label": "947", "value": "刺棘蓟"},
          {"label": "948", "value": "蘑菇"},
          {"label": "949", "value": "绿苹果"},
          {"label": "950", "value": "草莓"},
          {"label": "951", "value": "橘子"},
          {"label": "952", "value": "柠檬"},
          {"label": "953", "value": "无花果"},
          {"label": "954", "value": "菠萝"},
          {"label": "955", "value": "香蕉"},
          {"label": "956", "value": "菠萝蜜"},
          {"label": "957", "value": "蛋奶冻苹果"},
          {"label": "958", "value": "石榴"},
          {"label": "959", "value": "干草"},
          {"label": "960", "value": "烤面条加干酪沙司"},
          {"label": "961", "value": "巧克力酱,巧克力糖浆"},
          {"label": "962", "value": "面团"},
          {"label": "963", "value": "瑞士肉包,肉饼"},
          {"label": "964", "value": "披萨,披萨饼"},
          {"label": "965", "value": "馅饼"},
          {"label": "966", "value": "卷饼"},
          {"label": "967", "value": "红葡萄酒"},
          {"label": "968", "value": "意大利浓咖啡"},
          {"label": "969", "value": "杯子"},
          {"label": "970", "value": "蛋酒"},
          {"label": "971", "value": "高山"},
          {"label": "972", "value": "泡泡"},
          {"label": "973", "value": "悬崖"},
          {"label": "974", "value": "珊瑚礁"},
          {"label": "975", "value": "间歇泉"},
          {"label": "976", "value": "湖边,湖岸"},
          {"label": "977", "value": "海角"},
          {"label": "978", "value": "沙洲,沙坝"},
          {"label": "979", "value": "海滨,海岸"},
          {"label": "980", "value": "峡谷"},
          {"label": "981", "value": "火山"},
          {"label": "982", "value": "棒球,棒球运动员"},
          {"label": "983", "value": "新郎"},
          {"label": "984", "value": "潜水员"},
          {"label": "985", "value": "油菜"},
          {"label": "986", "value": "雏菊"},
          {"label": "987", "value": "杓兰"},
          {"label": "988", "value": "玉米"},
          {"label": "989", "value": "橡子"},
          {"label": "990", "value": "玫瑰果"},
          {"label": "991", "value": "七叶树果实"},
          {"label": "992", "value": "珊瑚菌"},
          {"label": "993", "value": "木耳"},
          {"label": "994", "value": "鹿花菌"},
          {"label": "995", "value": "鬼笔菌"},
          {"label": "996", "value": "地星"},
          {"label": "997", "value": "多叶奇果菌"},
          {"label": "998", "value": "牛肝菌"},
          {"label": "999", "value": "玉米穗"},
          {"label": "1000", "value": "卫生纸"},
        ];
      },
      handleSelect(row) {
        // console.log(row);
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
