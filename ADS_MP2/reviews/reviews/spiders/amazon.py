import scrapy


class Reviews(scrapy.Spider):

    # Spider name
    name: str = 'amazon'

    # Base URL instapot reviews
    myBaseUrl = "https://www.amazon.com/product-reviews/B00FLYWNYQ/ref=cm_cr_arp_d_viewopt_sr?pageNumber="
    start_urls = []

    # Creating list of urls to be scraped by appending page number at the end of base url
    for i in range(1, 501):
        start_urls.append(myBaseUrl + str(i))

    # Defining a Scrapy parser
    def parse(self, response):

        # Get the Review List
        data = response.css('#cm_cr-review_list')

        # Get the Name
        # name = data.css('.a-profile-name')

        # Get the Review Title
        # title = data.css('.review-title')

        # Get the Ratings
        stars = data.css('.review-rating')

        # Get the users Comments
        comments = data.css('.review-text')

        count = 0

        # combining the results
        for review in stars:

            yield {  # 'Name': ''.join(name[count].xpath(".//text()").extract()),
                # 'Title': ''.join(title[count].xpath(".//text()").extract()),
                'Rating': ''.join(review.xpath('.//text()').extract()),
                'Comment': ''.join(comments[count].xpath(".//text()").extract()).strip()
            }
            count = count + 1




