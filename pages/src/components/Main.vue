<script>
import Title from './sections/Title.vue'
import Abstract from './sections/Abstract.vue'
import Overview from './sections/Overview.vue'
import Detail from './sections/Detail.vue'
import Result from './sections/Result.vue'
import BibTeX from './sections/BibTeX.vue'
export default {
  components: {
    Title,
    Abstract,
    Overview,
	Detail,
	Result,
    BibTeX
  }
}
</script>

<template>
  <Title/>
  <Abstract/>
  <Overview/>
  <Detail/>
  <Result/>
  <BibTeX/>
</template>